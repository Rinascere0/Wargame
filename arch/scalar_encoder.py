import sys

sys.path.append('..')
import numpy as np
import torch
from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import to_one_hot,to_bin
from torch import nn
import pickle

debug=False

class Scalar_encoder(nn.Module):
    def __init__(self, device='cpu'):
        super(Scalar_encoder, self).__init__()
        self.fc = nn.Linear(HP.scalar_size, HP.scalar_embedding_size)
        self.relu = nn.ReLU(inplace=True)

        self.debug = debug
        self.device = device

    def forward(self, obs):
        x=obs.to(self.device)
        out = self.relu(self.fc(x))
        return out

    def preprocess(self, obs):
        scalar_list = []
        scalar_list.append(to_bin(obs['time']['cur_step'],11))


        for city in obs['cities']:
            color = city['flag']
            city_occu = np.zeros((2))
            if color<2:
                city_occu[color]=1
            scalar_list.append(city_occu)

        scalar_tensor = torch.tensor(scalar_list, dtype=torch.float32).reshape(1, -1)
        return scalar_tensor


if __name__ == '__main__':
    encoder = Scalar_encoder(debug=True).cuda()
    with open('../lib/obs.pkl', 'rb') as f:
        obs = pickle.load(f)

    scalar = encoder(obs[0])
    print(scalar.shape)
