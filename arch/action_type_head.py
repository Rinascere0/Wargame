import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *

debug=False

class ActionTypeHead(nn.Module):
    def __init__(self, device='cpu'):
        super(ActionTypeHead, self).__init__()
        self.fc1 = nn.Linear(HP.embedding_size, HP.original_256)
        self.fc2 = nn.Linear(HP.entity_embedding_size, HP.original_256)
        self.fc3 = nn.Linear(HP.autoregressive_embedding_size, HP.original_256)
        self.fc4 = nn.Linear(HP.original_256, HP.action_num)
        self.fc5 = nn.Linear(HP.action_num, HP.autoregressive_embedding_size)

        self.conv = nn.Conv1d(in_channels=HP.embedding_size, out_channels=HP.original_32, kernel_size=1,
                              stride=1, padding=False)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.device=device
        self.debug = debug

    def forward(self, autoregressive_embedding, state, unit_embedding, action_type_mask):
        x = F.relu(self.fc1(state))
        x = x + self.relu(self.fc2(unit_embedding))
        x = x + self.relu(self.fc3(autoregressive_embedding))

        action_type_logits = self.fc4(x)

        action_type_mask=action_type_mask.to(self.device)
        masked_action_logits = action_type_logits + (~action_type_mask.bool()) * (-1e9)

        action_prob = self.softmax(masked_action_logits)

        dist = Categorical(action_prob)
        action_type = dist.sample()
        log_prob = dist.log_prob(action_type)

        action_type_one_hot = to_one_hot_2(action_type, HP.action_num).to(self.device)
        action_type_one_hot = action_type_one_hot.squeeze(-2)

        autoregressive_embedding =  autoregressive_embedding+self.fc5(action_type_one_hot)

        if self.debug:
            print('action_type', action_type)
            print('action_type_logits', masked_action_logits)

    #    with open('logits.txt', 'a') as f:
    #        f.write(str(masked_action_logits) + '\n')
        return action_type, log_prob, autoregressive_embedding

    def evaluate(self, autoregressive_embedding, state, unit_embedding, action_type_mask, action_type):
        x = F.relu(self.fc1(state))
        x = x + self.relu(self.fc2(unit_embedding))
        x = x + self.relu(self.fc3(autoregressive_embedding))

        action_type_logits = self.fc4(x)

        action_type_mask=action_type_mask.to(self.device)
        masked_action_logits = action_type_logits + (~action_type_mask.bool()) * (-1e9)

        action_prob = self.softmax(masked_action_logits)
        dist = Categorical(action_prob)
        entropy = dist.entropy()
        log_prob = dist.log_prob(action_type)

        action_type_one_hot = to_one_hot_2(action_type, HP.action_num).to(self.device)
        action_type_one_hot = action_type_one_hot.squeeze(-2)

        autoregressive_embedding =  autoregressive_embedding+self.fc5(action_type_one_hot)

        return entropy, log_prob, autoregressive_embedding


if __name__ == '__main__':
    head = ActionTypeHead()
    x = torch.zeros([3, HP.embedding_size], dtype=torch.float32)
    action, logits, autoregressive_embedding = head(x)
    print(action)
    print(logits)
    print(autoregressive_embedding.shape)
