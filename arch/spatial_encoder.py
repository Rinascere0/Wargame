import torch
from torch import nn
import sys
import pickle

sys.path.append('..')

from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *

debug=False


class Spatial_encoder(nn.Module):
    def __init__(self, color,n_Resblocks=1,device='cpu'):
        super(Spatial_encoder, self).__init__()

        self.project = torch.nn.Conv2d(in_channels=HP.total_map_channel, out_channels=16, kernel_size=1,
                                       stride=1, padding=0, bias=False)
        self.ds1 = torch.nn.Conv2d(in_channels=HP.original_32, out_channels=HP.original_64, kernel_size=4, stride=2, padding=1,
                                   bias=False)
        self.ds2 = torch.nn.Conv2d(in_channels=HP.original_64, out_channels=HP.original_128, kernel_size=4, stride=2, padding=1,
                                   bias=False)
        self.ds3 = torch.nn.Conv2d(in_channels=HP.original_128, out_channels=HP.original_128, kernel_size=4, stride=2, padding=1,
                                   bias=False)
        self.resblocks = nn.ModuleList([
            ResBlock(inplanes=HP.original_128, planes=HP.original_128, stride=1) for _ in range(n_Resblocks)
        ])

        self.fc = torch.nn.Linear(16 * 16 * HP.original_128, HP.spatial_embedding_size)

        self.relu = nn.ReLU(inplace=True)

        self.map_tensor = None
        
        self.color=color
        
        self.debug = debug
        self.device = device

    def init_info(self, map_data):
        mapx, mapy = len(map_data), len(map_data[0])
        self.map_tensor = torch.zeros([128, 128, HP.map_channel])

        for i in range(mapx):
            for j in range(mapy):
                pos = map_data[i][j]
                pos_list = []
                pos_list.append(to_one_hot(pos['cond'], 6))
                pos_list.append(to_one_hot(pos['roads'], 4))
                pos_list.append(to_one_hot(pos['rivers'], 2))
                pos_list.append(torch.tensor([pos['elev'] / 100], dtype=torch.float32).reshape(1, -1))

                pos_tensor = torch.cat(pos_list, dim=1)
                self.map_tensor[i][j] = pos_tensor

        self.map_tensor = self.map_tensor.permute(2, 0, 1).unsqueeze(0)

    def pre_process(self, obs):
        unit_map = torch.zeros(128,128, 2)
        for unit in obs['operators']:
            unit_pos = unit['cur_hex']
            if unit['color']==self.color:
                unit_map[unit_pos // 100, unit_pos % 100, 0] = 1
            else:
                unit_map[unit_pos // 100, unit_pos % 100, 0] = -1
        for city in obs['cities']:
            city_pos = city['coord']
            unit_map[city_pos // 100, city_pos % 100, 1] = 1
        return unit_map.permute(2, 0, 1).unsqueeze(0)

    def forward(self, spatial_prep):
        batch_size = spatial_prep.shape[0]
        x = torch.cat([self.map_tensor.repeat(batch_size, 1, 1, 1), spatial_prep], dim=1).to(
            self.device)
        if self.debug:
            print('map_tensor:', x.shape)
        out = self.relu(self.project(x))
        if self.debug:
            print('project shape:', out.shape)

        out = self.ds1(out)
        if self.debug:
            print('ds1 shape:', out.shape)

        out = self.ds2(out)
        if self.debug:
            print('ds2 shape:', out.shape)

        out = self.ds3(out)
        if self.debug:
            print('ds3 shape:', out.shape)

        map_skip = out
        for resblock in self.resblocks:
            out = resblock(out)
            map_skip = map_skip + out

        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        spatial_embedding = self.relu(out)
        
        del out

        return map_skip, spatial_embedding


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x + z


if __name__ == '__main__':
    encoder = Spatial_encoder(debug=True, device='cuda').cuda()
    with open('../lib/map.pkl', 'rb') as f:
        map = pickle.load(f)
    with open('../lib/obs.pkl', 'rb') as f:
        obs = pickle.load(f)
    encoder.init_info(map)
    prep = encoder.pre_process(obs[0])
    embed, map_skip = encoder(prep)
    print('embed shape:', embed.shape)
    print('skip shape:', map_skip.shape)
