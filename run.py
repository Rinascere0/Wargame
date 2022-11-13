from learn.actor import Actor
from learn.learner import Learner
import os
import argparse
import pickle
import torch
from arch.arch_model import ArchModel

from ai.map import Map
path=os.path.dirname( os.path.abspath(__file__))

actor_num = 1
learner_num = 1

RED,BLUE= 0,1
color=RED

load_red=True
load_blue=False

device='cuda:6'

scenario=201033029601
test=False

class Run:
    def __init__(self): 
        models=[ArchModel(color=RED,device=device).to(device),ArchModel(color=BLUE,device=device).to(device)]
        print(os.path.exists('model_save/'+str(scenario)+'/model_red.pth'))
        if load_red and os.path.exists('model_save/'+str(scenario)+'/model_red.pth'):
            models[RED].load_state_dict(torch.load('model_save/'+str(scenario)+'/model_red.pth'))
            print('Loaded red model')
        if load_blue and os.path.exists('model_save/'+str(scenario)+'/model_blue.pth'):
            models[BLUE].load_state_dict(torch.load('model_save/'+str(scenario)+'model_blue.pth'))
            print('Loaded blue model')
        map_info= Map(scenario=scenario, data_dir='train_env/Data')  
        models[0].init_info(map_info)
        models[1].init_info(map_info)
        
        self.actors = [Actor(actor_id,scenario,map_info,models[color],models[1-color],test) for actor_id in range(actor_num)]
        self.learners=[ Learner(scenario,models[color],path,device)]
        for actor in self.actors:
            actor.set_learner(self.learners[0])
            
        
    def start(self):
        for learner in self.learners:
            learner.start()

        for actor in self.actors:
            actor.start()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--game_num', type=int, default=5000, metavar='N',
                        help='max_game_num (default: 1000')
    parser.add_argument('--batch_size', type=int, default=128, metavar='B',
                        help='batch_size (default: 128')
    parser.add_argument('--scenario', type=int, default=201033029601, metavar='S',
                        help='scenario (default: 201033029601')
    parser.add_argument('--color', type=int, default=RED, metavar='C',
                        help='color (default: RED)')
    parser.add_argument('--mode', type=str, default='train', metavar='M',
                        help='mode (default: train)')
    parser.add_argument('--refresh', type=bool, default=False, metavar='Rf',
                        help='refresh')
    parser.add_argument('--debug', type=bool, default=False, metavar='db',
                        help='debug')
    args = parser.parse_args()
    run = Run()
    run.start()
