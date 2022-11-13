import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *
from lib.const import ActionType
from torch.nn import functional as F
from torch.distributions import Categorical

debug = False

max_selected = HP.max_entity / 2


class SelectUnitHead(nn.Module):
    def __init__(self, device='cpu'):
        super(SelectUnitHead, self).__init__()

        self.conv = nn.Conv1d(in_channels=HP.embedding_size,
                              out_channels=HP.original_32, kernel_size=1, stride=1,
                              padding=0, bias=True)
        self.fc1 = nn.Linear(HP.autoregressive_embedding_size, HP.original_256)
        self.fc2 = nn.Linear(HP.original_256, HP.original_32)
        self.lstm = torch.lstm()

        self.fc3 = nn.Linear()

        self.device = device

    # TODO: implement select unit mask ,decided from action_type and select_target
    def forward(self, autoregressive_embedding, entity_embedding, select_unit_mask, unit_num):
        batch_size = entity_embedding.shape[0]

        entity_embedding = entity_embedding.reshape(batch_size, HP.max_entity, HP.entity_embedding_size)
        key = self.conv(entity_embedding)
        key_avg = torch.sum(key, dim=1) / unit_num.reshape(batch_size, 1)

        unit_logits = []
        units = []
        hidden = None

        is_end = torch.zeros(batch_size, device=self.device).bool()
        end_idx = HP.max_entity

        mask = torch.arange(HP.entity_size, device=self.device).float()
        mask = mask.repeat(batch_size, 1)
        mask[torch.arange(batch_size), end_idx] = False

        select_units_num = torch.ones(batch_size, device=self.device) * max_selected
        query = self.fc1(autoregressive_embedding)

        units = []
        total_logits = []
        for i in range(max_selected):
            if i == 1:
                mask[torch.arange(torch.arange(batch_size), end_idx)] = True

            x = self.fc1(autoregressive_embedding)
            x = F.relu(x)

            query, hidden = self.lstm(x, hidden)
            y = torch.sum(query * key, dim=-1)

            unit_logits = y.masked_fill(~mask, -1e9)
            unit_probs = F.softmax(x, dim=-1)

            dist = Categorical(unit_probs)
            unit_id = dist.sample()
            log_prob = dist.log_prob(unit_id)

            total_logits.append(unit_logits)
            units.append(unit_id)

            mask[torch.arange(batch_size), unit_id.squeeze(dim=1)] = False
            last_idx = (unit_id.squeeze(dim=1) == end_idx)

            is_end[last_idx] = 1

            select_units_num[last_idx] = i

            unit_one_hot = to_one_hot_2(unit_id, HP.max_entity).unsqueeze(-2)
            out = torch.bmm(unit_one_hot, key).squeeze(-2)

            out = out - key_avg

            t = self.project(out)
            autoregressive_embedding = autoregressive_embedding + t * ~is_end.unsqueeze(dim=1)

            if is_end.all():
                break

        return total_logits, units, autoregressive_embedding, select_units_num

    def evaluate(self):
        pass
