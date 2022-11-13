import time
import argparse
import pickle
import torch
import threading

from train_env import TrainEnv
from ai.demo_agent import DemoAgent
from ai.agent import Agent
from learn.learner import ReplayBuffer, PPO

from lib.functions import *

RED, BLUE, GREEN = 0, 1, -1
team = {BLUE: 'blue',
        RED: 'red'}
win_loss = {True: '_win',
            False: '_loss'}

skip_round = 3
debug = False

use_demo = True
write_action_log = True
use_local_reward = True

max_game_num = 100000

total_step = 1800


class Actor:
    def __init__(self, actor_id, scenario, map_info, model, foe_model, test):
        self.thread = threading.Thread(target=self.run, args=())
        self.learner = None
        self.model = model
        self.foe_model = foe_model

        self.color = model.color

        self.scenario = scenario

        self.actor_id = actor_id
        self.is_start = False
        self.is_running = False
        self.test = test

        self.map = map_info
        self.occupy_status = {}

    def start(self):
        self.is_start = True
        self.thread.start()

    def set_learner(self, learner):
        self.learner = learner

    def run(self):
        # instantiate agents and env
        env = TrainEnv()

        myAgent = Agent(self.scenario, self.model)
        if use_demo:
            foeAgent = DemoAgent()
        else:
            foeAgent = Agent(self.scenario, self.foe_model)

        train_num = 10
        for i in range(max_game_num):
            begin = time.time()
            self.occupy_status = {}
            game_id = f'{time.strftime("%Y-%m-%d-%H-%M-%S")}_{self.scenario}_{i}'
            trajectories = []
            print('>>> Actor ', self.actor_id, ', Game ', i)

            # setup everything
            state = env.setup({"scenario": self.scenario})
            print('Environment is ready.')
            myAgent.setup({
                "scenario": self.scenario,
                "seat": 1,
                "faction": self.color,
                "role": 0,
                "user_name": "agent",
                "user_id": 0,
                "state": state})
            foeAgent.setup({
                "scenario": self.scenario,
                "seat": 11,
                "faction": 1 - self.color,
                "role": 0,
                "user_name": "demo",
                "user_id": 0,
                "state": state})
            print('agents are ready.')

            # env gets myAgent_info, mandatory

            player_info = self.gen_myAgent_info(myAgent, foeAgent)
            state, done = env.env_config([player_info])
            print("player info configed")

            # 所有agent做战前部署
            deployment_actions = myAgent.deploy(observation=state[self.color])
            deployment_actions += foeAgent.deploy(observation=state[1 - self.color])
            # env handle deployments actions, optional
            state, done = env.env_config(deployment_actions)

            # init dynamic_mask
            self.model.init_mask(state)

            # loop until the end of game
            step = 0
            active_state = state
            reward = 0

            reward_list=[0,0,0,0,0]

            while not done:
                logits = None
                if step % skip_round == 0 and len(state[self.color]['operators']) != 0:
                    myAction, action, logits, value, prep = myAgent.step(state[self.color])
                    #    with open('obs.pkl', 'wb') as f:
                    #        pickle.dump(state[self.color], f)

                    if not use_demo:
                        foeAction, _, _, _, _ = foeAgent.step(state[1 - self.color])
                else:
                    myAction = []
                    foeAction = []

                if use_demo:
                    foeAction = foeAgent.step(state[1 - self.color])

                while None in myAction:
                    myAction.remove(None)
                while None in foeAction:
                    foeAction.remove(None)

                # output action
                if len(myAction + foeAction) > 0:
                    self.action_log(myAction, foeAction, step)

                # step env
                new_state, done = env.step(myAction + foeAction)

                # for test
                step += 1
                if step % 200 == 0:
                    print('Actor ', self.actor_id, ', Step ', step)

                # save trajectory
                if logits is not None:
                    if use_local_reward and (len(trajectories) > 0):
                        reward_add = self.gen_reward(active_state, state, [],reward_list)
                        trajectories[-1]['reward'] += reward_add
                        reward = self.gen_reward(state, new_state, myAction, reward_list)
                    trajectories.append({'state': prep,
                                         'action': action,
                                         'log_prob': logits,
                                         'reward': reward,
                                         'value': value,
                                         'is_terminal': done})
                    active_state = new_state

                state = new_state

            # game end
            win = new_state[self.color]['scores'][team[self.color] + '_win']

            if use_local_reward and logits is None:
                reward_add = self.gen_reward(active_state, state, [],reward_list)
                trajectories[-1]['reward'] += reward_add


            final_reward = 2 * (int(win > 0) - 0.5)

            # when counter dummy
            # final_reward=2*(int(win>150)-0.5)

            trajectories[-1]['reward'] += final_reward
            reward_list[0]+=final_reward
            
            if self.test:
                env.save_replay(game_id + win_loss[win > 0])
            else:
                self.learner.send_traj(trajectories, win, reward_list)

            with open('log/' + str(self.scenario) + '/win_loss.txt', 'a') as f:
                f.write(str(win) + '\n')

            # reset
            env.reset()
            myAgent.reset()
            foeAgent.reset()
            print(f'Total time of {game_id}: {time.time() - begin:.3f}s')

        print('Finish all games.')

    def action_log(self, myAction, foeAction, step):
        if write_action_log:
            with open('action' + str(self.actor_id) + '.txt', 'a') as f:
                f.write('Step' + str(step) + '\n');
                if len(myAction) > 0:
                    f.write('myAction:' + str(myAction) + '\n')
                if len(foeAction) > 0:
                    f.write('foeAction:' + str(foeAction) + '\n')
        return

    def gen_reward(self, state, new_state, actions,reward_list):
        atk_mul = 0.01
        lost_mul = 0
        occupy_mul = 0.1
        new_scores = new_state[self.color]['scores']
        old_scores = state[self.color]['scores']
        score_reward = new_scores[team[self.color] + '_win'] - old_scores[team[self.color] + '_win']

        # atk 
        attack_reward = (new_scores[team[self.color] + '_attack'] - old_scores[team[self.color] + '_attack']) * atk_mul

        # lost
        lost_reward = (new_scores[team[self.color] + '_remain'] - old_scores[team[self.color] + '_remain']) * lost_mul

        # move
        move_reward = 0
        old_pos = {}
        city_pos = get_city_pos(state[self.color], self.color)
        for unit in state[self.color]['operators']:
            if unit['color'] == self.color:
                old_pos[unit['obj_id']] = unit['cur_hex']

        min_dist = 1000
        for action in actions:
            if action['type'] == ActionType.Move:
                try:
                    new_pos = action['move_path'][-1]
                except:
                    with open('1.txt', 'w') as f:
                        f.write(str(action))
                unit_id = action['obj_id']
                if unit_id in old_pos:
                    for city in city_pos:
                        old_dist = self.map.get_distance(old_pos[unit_id], city)
                        new_dist = self.map.get_distance(new_pos, city)
                        min_dist = min(min_dist, old_dist)
                        if new_dist < old_dist:
                            move_reward += 0.02
                            break

    #    if state[self.color]['time']['cur_step'] <= total_step * 0.5 and min_dist >= 5 and move_reward == 0:
    #        move_reward = -0.02

        # occupy
        occupy_reward = 0
        for city in new_state[self.color]['cities']:
            coord = city['coord']
            if coord in self.occupy_status:
                if city['flag'] == self.color and not self.occupy_status[coord]:
                    self.occupy_status[coord] = True
                    occupy_reward += 0.05
                if city['flag'] != self.color and self.occupy_status[coord]:
                    self.occupy_status[coord] = False
                    occupy_reward -= 0
            else:
                self.occupy_status[coord] = True
                occupy_reward += 0.1
        
        
        total_reward=attack_reward + occupy_reward + lost_reward + move_reward
        
        reward_list[0]+=total_reward
        reward_list[1]+=attack_reward
        reward_list[2]+=lost_reward
        reward_list[3]+=occupy_reward
        reward_list[4]+=move_reward
        
        return total_reward

    def gen_myAgent_info(self, *argv):
        """
        generate myAgent_info from agents
        """''
        result = dict()
        result["type"] = 102
        info = dict()
        result["info"] = info
        for arg in argv:
            info[arg.seat] = {"faction": arg.faction,
                              "role": arg.role,
                              "user_id": arg.user_id,
                              "user_name": arg.user_name}
        print(result)
        return result
