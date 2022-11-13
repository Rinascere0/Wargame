    import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *
from lib.const import ActionType
from torch.nn import functional as F
from torch.distributions import Categorical


class TargetHead(nn.Module):
    def __init__(self, device='cpu'):
        super(TargetHead, self).__init__()
        self.fc1 = nn.Linear(HP.autoregressive_embedding_size, HP.original_32)
        self.fc2 = nn.Linear(HP.original_256, HP.original_32)
        self.fc3 = nn.Linear(HP.max_entity, HP.autoregressive_embedding_size)

        self.conv = nn.Conv1d(in_channels=HP.entity_embedding_size, out_channels=HP.original_32, kernel_size=1,
                              stride=1, padding=False)

        self.device=device
        self.debug = debug

    def evaluate(self, autoregressive_embedding, entity_embedding, action_type, load_masks, dload_masks, fire_masks,
                 target_unit):
        mask = (action_type == ActionType.GetOn) * load_masks + (action_type == ActionType.GetOff) * dload_masks + (
                action_type == ActionType.Shoot) * fire_masks

        entity_embedding = entity_embedding.reshape(-1, HP.max_entity, HP.entity_embedding_size)
        key = self.conv(entity_embedding.transpose(-1, -2)).transpose(-1, -2)
        if self.debug:
            print('key shape:', key.shape)

        query = self.fc1(autoregressive_embedding)
        query = query.unsqueeze(1)

        if self.debug:
            print('query shape:', query.shape)

        y = torch.bmm(key, query.transpose(-1, -2))

        y = y.squeeze(-1)
        if self.debug:
            print('y shape:', y.shape)
        target_unit_logits = y + (-1e9) * (~mask.bool())

        target_unit_mask = action_involve_target_mask(action_type).bool()

        target_unit_prob = F.softmax(target_unit_logits, dim=-1)
        dist = Categorical(target_unit_prob)
        log_prob = dist.log_prob(target_unit.squeeze())

        entropy = dist.entropy()

        # gen one hot
        target_unit_one_hot = to_one_hot_2(target_unit, HP.max_entity)

        autoregressive_embedding = autoregressive_embedding + F.relu(self.fc3(target_unit_one_hot))
        return entropy, log_prob, autoregressive_embedding

    def forward(self, autoregressive_embedding, entity_embedding, action_type, load_masks, dload_masks, fire_masks):
        mask = (action_type == ActionType.GetOn) * load_masks + (action_type == ActionType.GetOff) * dload_masks + (
                action_type == ActionType.Shoot) * fire_masks

        entity_embedding = entity_embedding.reshape(-1, HP.max_entity, HP.entity_embedding_size)
        key = self.conv(entity_embedding.transpose(-1, -2)).transpose(-1, -2)
        if self.debug:
            print('key shape:', key.shape)

        query = self.fc1(autoregressive_embedding)
        query = query.unsqueeze(1)

        if self.debug:
            print('query shape:', query.shape)

        y = torch.bmm(key, query.transpose(-1, -2))

        y = y.squeeze(-1)
        if self.debug:
            print('y shape:', y.shape)
        target_unit_logits = y + (-1e9) * (~mask.bool())

        target_unit_mask = action_involve_target_mask(action_type).bool()

        target_unit_prob = F.softmax(target_unit_logits, dim=-1)
        dist = Categorical(target_unit_prob)
        target_unit = dist.sample()
        log_prob = dist.log_prob(target_unit)

        target_unit = target_unit.unsqueeze(dim=1)

        target_unit_logits = target_unit_logits * target_unit_mask

        if self.debug:
            print('target_logits:', target_unit_logits)
            print('target_unit:', target_unit)

        # gen one hot
        target_unit_one_hot = to_one_hot_2(target_unit, HP.max_entity)

        autoregressive_embedding = autoregressive_embedding + F.relu(self.fc3(target_unit_one_hot))
        return target_unit, log_prob, autoregressive_embedding


if __name__ == '__main__':
    pass
