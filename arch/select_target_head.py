import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *
from lib.const import ActionType
from torch.nn import functional as F
from torch.distributions import Categorical


class SelectTargetHead(nn.Module):
    def __init__(self,device='cpu'):
        super(SelectTargetHead,self).__init__()
        
        self.fc1 = nn.Linear(HP.autoregressive_embedding_size, HP.original_32)
        self.fc2 = nn.Linear(HP.entity_embedding_size, HP.original_256)

        self.fc3 = nn.Linear(HP.max_entity, HP.autoregressive_embedding_size)
        self.conv = nn.Conv1d(in_channels=HP.entity_embedding_size, out_channels=HP.original_32, kernel_size=1,
                              stride=1, padding=False)
        
        self.device=device

    def forward(self, autoregressive_embedding, entity_embedding, action_type,target_mask):
    
        batch_size=autoregressive_embedding.shape[0]
        
        entity_embedding = entity_embedding.reshape(batch_size, HP.max_entity, HP.entity_embedding_size)

        key = self.conv(entity_embedding.transpose(-1, -2)).transpose(-1, -2)
 
        query = self.fc1(autoregressive_embedding)
        query = query.unsqueeze(1)
        
        y = torch.bmm(key, query.transpose(-1, -2))

        y = y.squeeze(-1)
                
        target_mask = target_mask.to(self.device)

        select_target_logits = y + (~target_mask.bool()) * (-1e9)

        target_prob = F.softmax(select_target_logits,dim=-1)

        
        dist = Categorical(target_prob)
        select_target = dist.sample()
        log_prob = dist.log_prob(select_target)
        
        involve_mask=action_involve_target_mask(action_type).to(self.device).float()

        log_prob=log_prob*involve_mask

        target_one_hot = to_one_hot_2(select_target, HP.max_entity).to(self.device)
        target_one_hot = target_one_hot.squeeze(-2)

        autoregressive_embedding = autoregressive_embedding + self.fc3(target_one_hot)

        return select_target, log_prob, autoregressive_embedding

    def evaluate(self, autoregressive_embedding, entity_embedding, target_mask, action_type,select_target):
        batch_size=autoregressive_embedding.shape[0]
        
        entity_embedding = entity_embedding.reshape(batch_size, HP.max_entity, HP.entity_embedding_size)

        key = self.conv(entity_embedding.transpose(-1, -2)).transpose(-1, -2)
 
        query = self.fc1(autoregressive_embedding)
        query = query.unsqueeze(1)
        
        y = torch.bmm(key, query.transpose(-1, -2))

        y = y.squeeze(-1)

        target_mask = target_mask.to(self.device)
        select_target_logits = y+ (~target_mask.bool()) * (-1e9)

        target_prob = F.softmax(select_target_logits,dim=-1)

        dist = Categorical(target_prob)
        log_prob = dist.log_prob(select_target)
        entropy = dist.entropy()
        
        involve_mask=action_involve_target_mask(action_type).to(self.device).float()
        print('mask:',involve_mask.shape)
        print('prob:',log_prob.shape)
        
        log_prob=log_prob*involve_mask
        entropy=entropy*involve_mask
        
        target_one_hot = to_one_hot_2(select_target, HP.max_entity).to(self.device)
        target_one_hot = target_one_hot.squeeze(-2)

       
        autoregressive_embedding = autoregressive_embedding + self.fc3(target_one_hot)

        return entropy, log_prob, autoregressive_embedding
