import sys

sys.path.append("..")
import numpy as np
import torch
from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *
from torch import nn
from lib.transformer import Transformer
import pickle

debug=False

RED,BLUE=0,1

class Entity_encoder(nn.Module):
    def __init__(self,  color,device='cpu'):
        super(Entity_encoder, self).__init__()
        self.max_mapx = HP.mapx
        self.max_mapy = HP.mapy
        self.max_bop_type = 3
        self.max_bop_subtype = 8
        self.max_armor_type = 5
        self.max_move_state = 6
        self.max_tire = 3
        self.bias_value = HP.bias_value

        self.max_entities = HP.max_entity

        self.embed = nn.Linear(HP.entity_size, HP.original_256)
        self.conv1 = nn.Conv1d(HP.original_256, HP.original_256, kernel_size=1, stride=1,
                               padding=0, bias=True)
        self.fc = nn.Linear(HP.original_256, HP.entity_embedding_size)
        self.relu = nn.ReLU(inplace=True)
        self.transformer = Transformer(d_model=HP.original_256,d_inner=HP.original_1024,n_layers=3,n_head=2,d_k=HP.original_128,d_v=HP.original_128,dropout=0.)
        self.fc1 = nn.Linear(HP.original_256, HP.original_256)

        self.debug = debug
        self.device = device
        
        self.color=color

    def forward(self, obs):
        x=obs.to(self.device)

        batch_size = x.shape[0]
        tmp_x = torch.mean(x, dim=2, keepdim=False)
        tmp_y = (tmp_x > self.bias_value + 1e3)
        entity_num = torch.sum(tmp_y, dim=1, keepdim=False)
        # entity_num=[batch_size]
        mask = torch.arange(0, HP.max_entity).float().to(self.device)
        mask = mask.repeat(batch_size, 1)

        # mask: [batch_size, max_entities]
        mask = mask < entity_num.unsqueeze(dim=1)

        mask_seq_len = mask.shape[-1]
        tran_mask = mask.unsqueeze(1)
        tran_mask = tran_mask.repeat(1, mask_seq_len, 1)

        if self.debug:
            print(tran_mask)

        out = self.relu(self.embed(x))
        out = self.transformer(out, tran_mask)
        entity_embedding = self.relu(self.conv1(out.transpose(1, 2)).transpose(1, 2))

        masked_out = out * mask.unsqueeze(dim=2)
        z = masked_out.sum(dim=1)
        z = z / entity_num.unsqueeze(dim=1)
        embedded_entity = self.relu(self.fc1(z))

        del x,z,out
        
        return entity_embedding, embedded_entity

    def preprocess(self, obs):
        units_info = obs['operators']+obs['passengers']
        entity_list = []
        my_entity_list=[]
        foe_entity_list=[]
        for unit_info in units_info:
            unit_feature = []

            unit_feature.append(to_one_hot(unit_info['color'], 2))
            unit_feature.append(to_one_hot(unit_info['cur_hex'] // 100, self.max_mapx))
            unit_feature.append(to_one_hot(unit_info['cur_hex'] % 100, self.max_mapy))
            unit_feature.append(to_one_hot(unit_info['type'] - 1, self.max_bop_type))
            unit_feature.append(to_one_hot(unit_info['sub_type'], self.max_bop_subtype))
            unit_feature.append(to_one_hot(unit_info['armor'], self.max_armor_type))
            unit_feature.append(to_one_hot(unit_info['move_state'], self.max_move_state))
            unit_feature.append(to_one_hot(unit_info['tire'], self.max_tire))
            unit_feature.append(to_one_hot(unit_info['target_state'], self.max_move_state))

            unit_feature.append(to_one_hot(unit_info['can_to_move'], 2))
            unit_feature.append(to_one_hot(unit_info['flag_force_stop'], 2))
            unit_feature.append(to_one_hot(unit_info['stop'], 2))
            unit_feature.append(to_one_hot(unit_info['on_board'], 2))
            unit_feature.append(to_one_hot(unit_info['weapon_unfold_state'], 2))
            unit_feature.append(to_one_hot(unit_info['keep'], 2))
            unit_feature.append(to_one_hot(unit_info['lose_control'], 2))
            unit_feature.append(to_one_hot(unit_info['guide_ability'], 2))
            unit_feature.append(to_one_hot(unit_info['A1'], 2))
            unit_feature.append(to_one_hot(unit_info['B1'], 2))
            unit_feature.append(to_one_hot(unit_info['stack'], 2))

            unit_feature.append(torch.tensor([
                unit_info['value'],
                unit_info['blood'],
                unit_info['max_blood'],
                unit_info['speed'],
                unit_info['basic_speed'],
                unit_info['C2'],
                unit_info['C3'],
                unit_info['keep_remain_time'],
                unit_info['move_to_stop_remain_time'],
                unit_info['get_on_remain_time'],
                unit_info['get_off_remain_time'],
                unit_info['tire_accumulate_time'],
                unit_info['weapon_cool_time'],
                unit_info['alive_remain_time'],
                unit_info['change_state_remain_time'],
                len(unit_info['see_enemy_bop_ids']),
            ], dtype=torch.float32).reshape(1, -1))

            entity_tensor = torch.cat(unit_feature, dim=1)
          #  entity_list.append(entity_tensor)
            if unit_info['color']==self.color:
                my_entity_list.append(entity_tensor)
            else:
                foe_entity_list.append(entity_tensor)
        
        entity_list=my_entity_list+foe_entity_list
        
        if len(entity_list) > 0:
            all_entity_tensor = torch.cat(entity_list, dim=0)
        else:
            all_entity_tensor = torch.zeros([0, HP.entity_size], dtype=torch.float32)

        self.real_entity_size = all_entity_tensor.shape[0]
        if self.real_entity_size < self.max_entities:
            bias_length = self.max_entities - all_entity_tensor.shape[0]
            bias = torch.zeros([bias_length, HP.entity_size])
            bias[:, :] = -1e9
            if self.debug:
                print('bias shape:', bias.shape)
            all_entity_tensor = torch.cat([all_entity_tensor, bias], dim=0)

        all_entity_tensor = all_entity_tensor.unsqueeze(0)
        if self.debug:
            print('entity_tensor shape:', all_entity_tensor.shape)

        return all_entity_tensor


if __name__ == '__main__':
    encoder = Entity_encoder(debug=True).cuda()
    with open('../lib/obs.pkl', 'rb') as f:
        obs = pickle.load(f)[0]

    embed = encoder(obs)
    print('embed shape:', embed[0].shape, embed[1].shape)
