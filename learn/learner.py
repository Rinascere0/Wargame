import sys

sys.path.append('..')

import torch
import copy
import threading
import traceback
import pickle
from tensorboardX import SummaryWriter
from time import time
import os
import argparse
from ai.map import Map

from torch import nn
from torch.nn import functional as Fnn
from torch.autograd import Variable
from arch.arch_model import ArchModel

path = os.path.dirname(os.path.abspath(__file__))

RED, BLUE = 0, 1

max_running_time = 24 * 10 * 60 * 60
batch_size = 256
learning_step = 60
test_step = 60

lr = 5e-5
betas = [0.9, 0.999]
gamma = 0.99
lamda = 0.99
k_epochs = 3
eps_clip = 0.2

use_reward_norm = False
debug = False
refresh = False
seperate_test = (learning_step != test_step)


class ReplayBuffer(object):
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        self.buffer_size = 0
        self.seperates = []

    def store_transition(self, trajectories):
        for trajectory in trajectories:
            self.actions.append(trajectory['action'])
            self.states.append(trajectory['state'])
            self.log_probs.append(trajectory['log_prob'])
            self.rewards.append(trajectory['reward'])
            self.values.append(trajectory['value'])
            self.is_terminals.append(trajectory['is_terminal'])

        self.buffer_size += len(trajectories)
        self.seperates.append(self.buffer_size)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]
        del self.seperates[:]
        self.buffer_size = 0


class Learner:
    def __init__(self, scenario, model, path, device):
        self.path = path
        self.model_path = path + '/model_save/' + str(scenario)
        self.model = model
        self.algo = PPO(model=self.model, path=self.model_path, device=device)

        #    self.thread = threading.Thread(target=self.run, args=())
        self.actors = []
        self.buffer = ReplayBuffer()
        self.logdir = 'log/' + str(scenario)
        self.writer = SummaryWriter(self.logdir)

        self.train_game_num = 0
        self.update_count = 0

        self.game_num = 0
        self.win_loss = [0, 0]
        self.win_rate = 0
        self.scores = [0, 0, 0, 0, 0, 0]

        self.learning_step = learning_step

        self.is_start = False
        self.is_running = False

        self.train = True

        if refresh:
            self.clear_log()

        if os.path.exists(self.logdir + '/win_loss.pkl'):
            with open(self.logdir + '/win_loss.pkl', 'rb') as f:
                self.win_loss = pickle.load(f)
                self.game_num = pickle.load(f)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        map_ = Map(scenario=scenario, data_dir=path + '/train_env/Data')
        self.model.init_info(map_)

    def clear_log(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        with open(self.logdir + '/win_loss.txt', 'w') as f:
            f.write('')
        with open(self.logdir + '/win_loss.pkl', 'wb') as f:
            pickle.dump(self.win_loss, f)
            pickle.dump(self.game_num, f)

    def start(self):
        self.is_start = True

    #     self.thread.start()

    def add_actor(self, actor):
        self.actors.append(actor)

    def send_traj(self, traj, score, reward_list):
        self.game_num += 1

        if not seperate_test or not self.train:
            self.win_loss[1 - int(score > 0)] += 1
            self.scores[0] += score
            for i in range(5):
                self.scores[i + 1] += reward_list[i]

            if self.game_num % test_step == 0:
                new_win_rate = self.win_loss[0] / test_step
                self.writer.add_scalar('win_rate', new_win_rate, self.update_count)
                self.writer.add_scalar('score', self.scores[0] / test_step, global_step=self.update_count)
                self.writer.add_scalar('reward', self.scores[1] / test_step, global_step=self.update_count)
                self.writer.add_scalar('attack', self.scores[2] / test_step, global_step=self.update_count)
                self.writer.add_scalar('lost', self.scores[3] / test_step, global_step=self.update_count)
                self.writer.add_scalar('occupy', self.scores[4] / test_step, global_step=self.update_count)
                self.writer.add_scalar('move', self.scores[5] / test_step, global_step=self.update_count)
                if new_win_rate > self.win_rate:
                    self.win_rate = new_win_rate
                    torch.save(self.model.state_dict(), self.model_path + '/model.pth')
                    torch.save(self.model.state_dict(), self.model_path + '/model_' + str(self.win_rate) + '.pth')
                self.win_loss = [0, 0]
                self.scores = [0, 0, 0, 0, 0, 0]
                #      if self.win_rate>=0.5:
                #          self.learning_step=learning_step*2

                if seperate_test:
                    self.game_num = 0
                    self.train = not self.train

        if self.train and self.game_num != 0:
            self.buffer.store_transition(traj)
            if self.game_num % self.learning_step == 0:
                loss = self.algo.update(self.buffer)
                self.buffer.clear()
                self.writer.add_scalar('loss', loss, global_step=self.update_count)
                self.update_count += 1

                self.game_num = 0
                if seperate_test:
                    self.train = not self.train

        with open(self.logdir + '/win_loss.pkl', 'wb') as f:
            pickle.dump(self.win_loss, f)
            pickle.dump(self.game_num, f)

    def run(self):
        start_time = time()
        while time() - start_time < max_running_time:
            try:
                actor_is_running = False
                for actor in self.actors:
                    if actor.is_running:
                        actor_is_running = True
                if not actor_is_running:
                    break
            except Exception as e:
                print("Learner.run() Exception cause break, Details of the Exception:", e)
                print(traceback.format_exc())
                break
            finally:
                self.is_running = False


class PPO(object):
    def __init__(self,
                 model=None,
                 path=None,
                 device='cuda:0'):

        self.model_path = path + '/model.pkl'
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lamda = lamda
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip

        self.model = model
        self.model_old = ArchModel(color=model.color, device=device)
        self.model_old.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=betas)
        self.criterion = nn.MSELoss()

        self.device = self.model.device

    def get_gaes(self, seperates, rewards, v_preds):
        total_gaes = []
        last_sep = 0
        v_pred_next = []
        for sep in seperates:
            if debug:
                print('reward', zip(rewards[last_sep:sep]))
                print('vp', v_preds[last_sep + 1:sep] + [0])
                print('v', v_preds[last_sep:sep])
            v_pred_next += v_preds[last_sep + 1:sep] + [0]
            deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in
                      zip(rewards[last_sep:sep], v_preds[last_sep + 1:sep] + [0], v_preds[last_sep:sep])]
            # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
            gaes = copy.deepcopy(deltas)
            for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
                gaes[t] = gaes[t] + self.gamma * self.lamda * gaes[t + 1]
            total_gaes += gaes
            last_sep = sep
        total_gaes = torch.tensor(total_gaes).to(self.device)
        total_gaes = (total_gaes - total_gaes.mean()) / (total_gaes.std() + 1e-10)
        return total_gaes, torch.tensor(v_pred_next).to(self.device)

    def get_logit_batch(self, logits, begin, end):
        select_unit_logits = []
        action_type_logits = []
        location_logits = []
        select_target_logits=[]
        for s, a, l, t in logits[begin:end]:
            select_unit_logits.append(s.to(self.device))
            action_type_logits.append(a.to(self.device))
            location_logits.append(l.to(self.device))
            select_target_logits.append(t.to(self.device))
        return torch.stack(select_unit_logits), torch.stack(action_type_logits), torch.stack(location_logits),torch.stack(select_target_logits)

    def get_action_batch(self, actions, begin, end):
        select_units = []
        action_types = []
        locations = []
        select_targets=[]
        for action in actions[begin:end]:
            select_unit, action_type, location ,select_target= action['select_unit'], action['action_type'], action['location'],action['target']
            select_units.append(select_unit.to(self.device))
            action_types.append(action_type.to(self.device))
            locations.append(location.to(self.device))
            select_targets.append(select_target.to(self.device))
        select_units = torch.stack(select_units, dim=0).squeeze(1)
        action_types = torch.stack(action_types, dim=0).squeeze(1)
        locations = torch.stack(locations, dim=0).squeeze(1)
        select_targets=torch.stack(select_targets, dim=0).squeeze(1)
        if debug:
            print("select_units", select_units.shape)
            print('action_types', action_types.shape)
            print('locations', locations.shape)
        return select_units, action_types, locations,select_targets

    def get_prep(self, preps, begin, end):
        entity_preps = []
        spatial_preps = []
        #   scalar_preps = []
        select_unit_masks = []
        action_type_masks = []
        select_target_masks = []
        masks = []
        for prep in preps[begin:end]:
            entity_prep, spatial_prep, mask = prep
            entity_preps.append(entity_prep)
            spatial_preps.append(spatial_prep)
            #      scalar_preps.append(scalar_prep)
            select_unit_mask, action_type_mask,select_target_mask = mask
            select_unit_masks.append(select_unit_mask)
            action_type_masks.append(action_type_mask)
            select_target_masks.append(select_target_mask)
        entity_preps = torch.stack(entity_preps, dim=0).squeeze(1)
        spatial_preps = torch.stack(spatial_preps, dim=0).squeeze(1)
        #  scalar_preps = torch.stack(scalar_preps, dim=0).squeeze(1)
        action_type_masks =torch.stack(action_type_masks,dim=0).squeeze(1)
        select_unit_masks = torch.stack(select_unit_masks, dim=0).squeeze(1)
        select_target_masks = torch.stack(select_target_masks, dim=0).squeeze(1)
        if debug:
            print('entity_prep', entity_preps.shape)
            print('spatial_prep', spatial_preps.shape)
            #     print('scalar_prep', scalar_preps.shape)
            print('select_unit_mask', select_unit_masks.shape)
        return entity_preps, spatial_preps, (select_unit_masks, action_type_masks, select_target_masks)

    def update(self, replay_buffer):
        #   with open('buffer.pkl','wb') as f:
        #        pickle.dump(replay_buffer,f)

        rewards = torch.tensor(replay_buffer.rewards).to(self.device)
        if use_reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_state_values = torch.stack(replay_buffer.values).squeeze(-1).squeeze(-1)
        values_list = old_state_values.detach().cpu().numpy().tolist()
        advantages, v_pred_nexts = self.get_gaes(replay_buffer.seperates, replay_buffer.rewards, values_list)

        buffer_size = replay_buffer.buffer_size
        batch_num = (buffer_size - 1) // batch_size + 1

        old_states = []
        old_actions = []
        old_log_probs = []
        v_pred_next_batches = []
        adv_batches = []
        reward_batches = []
        for batch_id in range(batch_num):
            begin, end = batch_id * batch_size, (batch_id + 1) * batch_size
            old_states.append(self.get_prep(replay_buffer.states, begin, end))
            old_actions.append(self.get_action_batch(replay_buffer.actions, begin, end))
            old_log_probs.append(self.get_logit_batch(replay_buffer.log_probs, begin, end))
            adv_batches.append(advantages[begin:end])
            reward_batches.append(rewards[begin:end])
            v_pred_next_batches.append(v_pred_nexts[begin:end])

        loss = []
        for epoch in range(self.k_epochs):
            print('epoch', epoch)
            for batch_id in range(batch_num):
                print('batch', batch_id)
                log_probs, dist_entropy, state_values = self.model.evaluate(old_states[batch_id], old_actions[batch_id])
                action_loss = 0
                for old_log_prob, log_prob in zip(old_log_probs[batch_id], log_probs):
                    ratios = torch.exp(log_prob - old_log_prob.squeeze(1))
                    surr1 = ratios * adv_batches[batch_id]
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_batches[batch_id]
                    action_loss += -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * self.criterion(state_values,
                                                  v_pred_next_batches[batch_id] + reward_batches[batch_id])
                entropy_loss = - 0.01 * torch.mean(dist_entropy)

                print('value loss=', value_loss)
                print('entropy loss=', entropy_loss)
                print('action loss=', action_loss)
                total_loss = value_loss + entropy_loss + action_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                loss.append(total_loss.item())
                del value_loss, entropy_loss, action_loss, log_probs, dist_entropy, state_values, total_loss

        self.model_old.load_state_dict(self.model.state_dict())

        del old_states, old_actions, old_log_probs, v_pred_next_batches, adv_batches, reward_batches

        loss_mean = sum(loss) / len(loss)
        print('Model updated, Loss=', loss_mean)

        return loss_mean


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
    path = path + '/..'
    learner = Learner(args, path)
    learner.start()

    with open(path + '/traj.pkl', 'rb') as f:
        buffer = pickle.load(f)
    learner.algo.update(buffer)
