import torch
import numpy as np
from lib.const import ActionType

def get_enemy_pos(obs,color):
    pos=[]
    for unit in obs['operators']:
        if unit['color']!=color:
            pos.append(unit['cur_hex'])
    return pos
    
def get_city_pos(obs,color=-1):
    pos=[]
    for city in obs['cities']:
        if color==-1 or city['flag']!=color:
            pos.append(city['coord'])
    return pos
        
def to_one_hot(x, max_size):
    out = torch.zeros([max_size])
    out[x] = 1
    return out.reshape(1, -1)

def to_one_hot_2(x, max_size):
    shape = x.shape
    one_hot = torch.zeros(shape[0], max_size)
    for i in range(shape[0]):
        one_hot[i][x[i]] = 1

    return one_hot


def action_involve_location_mask(action_type):
    mask = torch.zeros_like(action_type)
    for i, action in enumerate(action_type):
        mask[i] = action in [ActionType.Move, ActionType.JMPlan,ActionType.Ambush]
    return mask


def action_involve_target_mask(action_type):
    mask = torch.zeros_like(action_type)
    for i, action in enumerate(action_type):
        mask[i] = action in [ActionType.Shoot, ActionType.GetOn, ActionType.GetOff]
    return mask


def to_bin(x, max_size):
    out=np.zeros((max_size))
    i=max_size-1
    while (x>0):
        out[i]=x%2
        x/=2
        i-=1
    return torch.tensor(out).reshape(1,-1)


def to_bin_1(x, max_size):
    out = np.unpackbits(np.array([x],dtype = np.uint8))
    return torch.tensor(out,dtype=torch.float32).reshape(1,-1)
