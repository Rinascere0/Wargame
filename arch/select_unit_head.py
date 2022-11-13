import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *
from lib.const import ActionType
from torch.nn import functional as F
from torch.distributions import Categorical

debug=False
class SelectUnitHead(nn.Module):
    def __init__(self, device='cpu'):
        super(SelectUnitHead, self).__init__()
        self.fc1 = nn.Linear(HP.embedding_size, HP.original_32)
        self.fc2 = nn.Linear(HP.original_256, HP.original_32)
        self.fc3 = nn.Linear(HP.max_entity, HP.autoregressive_embedding_size)
        self.fc4 = nn.Linear(HP.embedding_size, HP.autoregressive_embedding_size)
        self.conv = nn.Conv1d(in_channels=HP.entity_embedding_size, out_channels=HP.original_32, kernel_size=1,
                              stride=1, padding=False)

        self.device=device
        self.debug = debug

    def forward(self, state, entity_embedding, mask):
        batch_size = entity_embedding.shape[0]

        entity_embedding = entity_embedding.reshape(batch_size, HP.max_entity, HP.entity_embedding_size)
        key = self.conv(entity_embedding.transpose(-1, -2)).transpose(-1, -2)
        if self.debug:
            print('key shape:', key.shape)

        query = self.fc1(state)
        query = query.unsqueeze(1)

        if self.debug:
            print('query shape:', query.shape)

        y = torch.bmm(key, query.transpose(-1, -2))

        y = y.squeeze(-1)
        if self.debug:
            print('y shape:', y.shape)
            
        mask=mask.to(self.device)
        select_unit_logits = y + (-1e9) * (~mask.bool())

        select_unit_prob = F.softmax(select_unit_logits, dim=-1)
        dist = Categorical(select_unit_prob)
        select_unit = dist.sample()
        log_prob = dist.log_prob(select_unit)
        
        # gen one hot
        select_unit_one_hot = to_one_hot_2(select_unit.unsqueeze(dim=1), HP.max_entity).to(self.device)

        autoregressive_embedding = F.relu(self.fc3(select_unit_one_hot))
        
        autoregressive_embedding = autoregressive_embedding + self.fc4(state)
        
        return select_unit, log_prob, autoregressive_embedding

    def evaluate(self, state, entity_embedding, mask, select_unit):
        batch_size = entity_embedding.shape[0]

        entity_embedding = entity_embedding.reshape(batch_size, HP.max_entity, HP.entity_embedding_size)
        key = self.conv(entity_embedding.transpose(-1, -2)).transpose(-1, -2)
        if self.debug:
            print('key shape:', key.shape)

        query = self.fc1(state)
        query = query.unsqueeze(1)

        if self.debug:
            print('query shape:', query.shape)

        y = torch.bmm(key, query.transpose(-1, -2))

        y = y.squeeze(-1)
        if self.debug:
            print('y shape:', y.shape)
            
        mask=mask.to(self.device)
        select_unit_logits = y + (-1e9) * (~mask.bool())

        select_unit_prob = F.softmax(select_unit_logits, dim=-1)
        dist = Categorical(select_unit_prob)

        log_prob = dist.log_prob(select_unit)

        entropy = dist.entropy()
        

        # gen one hot
        select_unit_one_hot = to_one_hot_2(select_unit, HP.max_entity).to(self.device)

        autoregressive_embedding = F.relu(self.fc3(select_unit_one_hot))
        
        autoregressive_embedding = autoregressive_embedding + self.fc4(state)
        return entropy, log_prob, autoregressive_embedding
