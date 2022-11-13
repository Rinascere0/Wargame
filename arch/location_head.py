import math
import pickle

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *
import numpy as np
from lib.const import ActionType

debug=False

minx,maxx=79,91
miny,maxy=28,49

class LocationHead(nn.Module):
    def __init__(self, device='cpu'):
        super(LocationHead, self).__init__()
        mmc = HP.max_map_channel
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(HP.embed_x + HP.embed_y, HP.autoregressive_embedding_size)
        self.ds = nn.Conv2d(in_channels=mmc + 1, out_channels=mmc, kernel_size=1)
        self.us1 = nn.ConvTranspose2d(in_channels=mmc, out_channels=int(mmc / 2), kernel_size=4, stride=2, padding=1)
        self.us2 = nn.ConvTranspose2d(in_channels=int(mmc / 2), out_channels=int(mmc / 8), kernel_size=4, stride=2,
                                      padding=1)
        self.us3 = nn.ConvTranspose2d(in_channels=int(mmc / 8), out_channels=1, kernel_size=4, stride=2,
                                      padding=1)
        #  self.us4 = nn.ConvTranspose2d(in_channels=int(mmc / 8), out_channels=1, kernel_size=4, stride=2,
        #                                padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.mapx = 128
        self.mapy = 128
        
        self.device=device
        self.debug = debug

    def init_params(self, mapx, mapy):
        self.mapx = mapx
        self.mapy = mapy

    def evaluate(self, autoregressive_embedding, map_skip, action_type, location_id):
        batch_size = action_type.shape[0]
        reshap_x, reshap_y = map_skip.shape[-2], map_skip.shape[-1]
        map_skip = map_skip.expand((batch_size, HP.original_128, reshap_x, reshap_y))
        autoregressive_embedding_ = autoregressive_embedding.reshape(batch_size, -1, reshap_x, reshap_y)

        x = torch.cat([map_skip, autoregressive_embedding_], dim=1)
        x = self.relu(self.ds(x))
        x = x + map_skip
        x = self.relu(self.us1(x))
        x = self.relu(self.us2(x))
        x = self.relu(self.us3(x))
        y = x.reshape(batch_size, self.mapx, self.mapy)

        target_location_logits = y
        target_location_logits = target_location_logits.reshape(batch_size, -1)

        mask = torch.zeros(self.mapx, self.mapy).bool().to(self.device)
        for i in range(minx,maxx+1):
            for j in range(miny,maxy+1):
                mask[i][j] = True

                
        mask = mask.reshape(1, -1)

        target_location_logits = target_location_logits + (~mask) * (-1e9)
        target_location_probs = self.softmax(target_location_logits)

        dist = Categorical(target_location_probs)
        log_prob = dist.log_prob(location_id)
        entropy = dist.entropy()
        
        location_mask = action_involve_location_mask(action_type).to(self.device).float()
        log_prob = log_prob*location_mask
        entropy=entropy*location_mask
        
        location_one_hot_list = []
        for location in location_id:
            location_one_hot_list.append(
                torch.cat(
                    [to_one_hot(location // HP.embed_y, HP.embed_x), to_one_hot(location % HP.embed_y, HP.embed_y)],
                    dim=1))

        location_one_hot = torch.cat(location_one_hot_list, dim=0).to(self.device)
        y = F.relu(self.fc2(location_one_hot))
        autoregressive_embedding = autoregressive_embedding + y

        return entropy, log_prob, autoregressive_embedding

    def forward(self, autoregressive_embedding, map_skip, action_type):
        batch_size = action_type.shape[0]
        reshap_x, reshap_y = map_skip.shape[-2], map_skip.shape[-1]
        map_skip = map_skip.expand((batch_size, HP.original_128, reshap_x, reshap_y))
        autoregressive_embedding_ = autoregressive_embedding.reshape(batch_size, -1, reshap_x, reshap_y)

        if self.debug:
            print('autoregressive_embedding_shape:', autoregressive_embedding_.shape)
            print('map_skip_shape', map_skip.shape)

        x = torch.cat([map_skip, autoregressive_embedding_], dim=1)
        if self.debug:
            print('x_shape:', x.shape)

        x = self.relu(self.ds(x))
        x = x + map_skip
        #  if self.debug:
        #      print(x.shape)

        x = self.relu(self.us1(x))
        #   if self.debug:
        #      print(x.shape)

        x = self.relu(self.us2(x))
        #   if self.debug:
        #       print(x.shape)

        x = self.relu(self.us3(x))
        #   if self.debug:
        #       print(x.shape)

        y = x.reshape(batch_size, self.mapx, self.mapy)
        #    if self.debug:
        #        print(y.shape)

        #    if self.debug:
        #        print('location_scale', location_scale)
        target_location_logits = y
        target_location_logits = target_location_logits.reshape(batch_size, -1)

        mask = torch.zeros(self.mapx, self.mapy).bool().to(self.device)
        for i in range(minx,maxx+1):
            for j in range(miny,maxy+1):
                mask[i][j] = True

        mask = mask.reshape(1, -1)

        target_location_logits = target_location_logits + (~mask) * (-1e9)
        target_location_probs = self.softmax(target_location_logits)

        dist = Categorical(target_location_probs)
        location_id = dist.sample()
        log_prob = dist.log_prob(location_id)

        location_mask = action_involve_location_mask(action_type).to(self.device).float()

        log_prob=log_prob*location_mask

        # gen_location_one_hot
        location_one_hot_list = []
        for location in location_id:
            location_one_hot_list.append(
                torch.cat(
                    [to_one_hot(location // HP.embed_y, HP.embed_x), to_one_hot(location % HP.embed_y, HP.embed_y)],
                    dim=1))

        location_one_hot = torch.cat(location_one_hot_list, dim=0).to(self.device)
        
        y = F.relu(self.fc2(location_one_hot))
        autoregressive_embedding = autoregressive_embedding + y
        
        del y

        return location_id, log_prob, autoregressive_embedding


if __name__ == '__main__':
    head = LocationHead()
    map_skip = torch.ones(1, 64, 8, 8)
    autoregressive_embedding = torch.ones(1, 256)
    action_type = torch.tensor([2])
    location_prob, location = head(map_skip, autoregressive_embedding, action_type)
    print(location_prob, location)
