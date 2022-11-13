from .base_agent import BaseAgent
from .map import Map
from lib.const import BopType, ActionType, MoveType
from lib.hyper_parameters import hyper_parameters as HP
from arch.arch_model import ArchModel
import pickle
import os
import random
import torch
import json

RED, BLUE = 0, 1


class AgentBase(BaseAgent):
    def __init__(self, scenario, model):
        super(AgentBase, self).__init__()
        self.scenario = scenario
        self.scenario_info = None
        self.color = model.color
        self.map = Map(scenario=self.scenario, data_dir='train_env/Data')
        self.game_num = 0
        self.map_data = self.map.get_map_data()
        self.observation = None
        self.unit_func_rev = {}

        #  self.path = 'model_save/' + str(self.scenario)
        #  if self.color == RED:
        #      self.model_path = self.path + '/red'
        #  else:
        #      self.model_path = self.path + '/blue'
        self.model_path = 'model_save/'

        self.gen_action = {
            ActionType.Occupy: self.gen_occupy,
            ActionType.Shoot: self.gen_shoot,
            ActionType.GuideShoot: self.gen_guide_shoot,
            ActionType.JMPlan: self.gen_jm_plan,
            ActionType.GetOn: self.gen_get_on,
            ActionType.GetOff: self.gen_get_off,
            ActionType.ChangeState: self.gen_change_state,
            ActionType.RemoveKeep: self.gen_remove_keep,
            ActionType.Move: self.gen_move,
            ActionType.StopMove: self.gen_stop_move,
            ActionType.WeaponLock: self.gen_WeaponLock,
            ActionType.WeaponUnFold: self.gen_WeaponUnFold,
            ActionType.CancelJMPlan: self.gen_cancel_JM_plan,
            ActionType.Ambush: self.gen_move,
        }

    def setup(self, setup_info):
        # normal setup
        self.scenario = setup_info["scenario"]
        self.get_scenario_info(setup_info["scenario"])
        self.color = setup_info["faction"]
        self.faction = setup_info["faction"]
        self.seat = setup_info["seat"]
        self.role = setup_info["role"]
        self.user_name = setup_info["user_name"]
        self.user_id = setup_info["user_id"]

    def step(self, observation):
        return

    def deploy(self, observation):
        self.team_info = observation["role_and_grouping_info"]
        self.controllable_ops = observation["role_and_grouping_info"][self.seat][
            "operators"
        ]
        communications = observation["communication"]
        for command in communications:
            if command["info"]["company_id"] == self.seat:
                if command["type"] == 200:
                    self.my_mission = command
                elif command["type"] == 201:
                    self.my_direction = command
        actions = []
        for item in observation["operators"]:
            if item["obj_id"] in self.controllable_ops:
                operator = item
                if operator["sub_type"] == 2 or operator["sub_type"] == 4:
                    actions.append(
                        {
                            "actor": self.seat,
                            "obj_id": operator["obj_id"],
                            "type": 303,
                            "target_obj_id": operator["launcher"],
                        }
                    )
        return actions

    def reset(self):
        self.scenario = None
        self.color = None
        self.scenario_info = None

    def get_scenario_info(self, scenario: int):
        #   SCENARIO_INFO_PATH = os.path.join(os.path.dirname(__file__), f'scenario_{scenario}.json')
        SCENARIO_INFO_PATH = 'train_env/Data/scenarios/' + f'{scenario}.json'
        with open(SCENARIO_INFO_PATH, encoding='utf8') as f:
            self.scenario_info = json.load(f)

    def get_bop(self, obj_id):
        """Get bop in my observation based on its id."""
        for bop in self.observation['operators']:
            if obj_id == bop['obj_id']:
                return bop

    def get_move_type(self, bop):
        bop_type = bop['type']
        if bop_type == BopType.Vehicle:
            if bop['move_state'] == MoveType.March:
                move_type = MoveType.March
            else:
                move_type = MoveType.Maneuver
        elif bop_type == BopType.Infantry:
            move_type = MoveType.Walk
        else:
            move_type = MoveType.Fly
        return move_type

    def gen_move(self, obj_id, target, candidate):
        bop = self.get_bop(obj_id)
        #    if bop['sub_type'] == 3:
        #        return
        destination = target
        if bop and bop['cur_hex'] != destination:
            move_type = self.get_move_type(bop)
            route = self.map.gen_move_route(
                bop['cur_hex'], destination, move_type)
            if len(route) == 0:
                with open('2.txt', 'w') as f:
                    f.write(str(bop['cur_hex']) + '\n' + str(destination))
            return {
                'actor': self.seat,
                'obj_id': obj_id,
                'type': ActionType.Move,
                'move_path': route,
            }

    def gen_occupy(self, obj_id, target, candidate):
        return {
            'actor': self.seat,
            'obj_id': obj_id,
            'type': ActionType.Occupy,
        }

    def gen_shoot(self, obj_id, target, candidate):
        if target is not None:
            candidate=[cand for cand in candidate if cand["target_obj_id"]==target]
        best = max(candidate, key=lambda x: x["attack_level"])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.Shoot,
            "target_obj_id": best["target_obj_id"],
            "weapon_id": best["weapon_id"],
        }

    def gen_guide_shoot(self, obj_id, target, candidate):
        best = max(candidate, key=lambda x: x['attack_level'])
        return {
            'actor': self.seat,
            'obj_id': obj_id,
            'type': ActionType.GuideShoot,
            'target_obj_id': best['target_obj_id'],
            'weapon_id': best['weapon_id'],
            'guided_obj_id': best['guided_obj_id'],
        }

    def gen_jm_plan(self, obj_id, target, candidate):
        weapon_id = random.choice(candidate)['weapon_id']
        jm_pos = target
        return {
            'actor': self.seat,
            'obj_id': obj_id,
            'type': ActionType.JMPlan,
            'jm_pos': jm_pos,
            'weapon_id': weapon_id,
        }

    def gen_get_on(self, obj_id, target, candidate):
        target_obj_id = random.choice(candidate)["target_obj_id"]
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.GetOn,
            "target_obj_id": target_obj_id,
        }

    def gen_get_off(self, obj_id, target, candidate):
        target_obj_id = random.choice(candidate)["target_obj_id"]
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.GetOff,
            "target_obj_id": target_obj_id,
        }

    def gen_change_state(self, obj_id, target, candidate):
        target_state = random.choice(candidate)['target_state']
        return {
            'actor': self.seat,
            'obj_id': obj_id,
            'type': ActionType.ChangeState,
            'target_state': target_state,
        }

    def gen_remove_keep(self, obj_id, target, candidate):
        return {
            'actor': self.seat,
            'obj_id': obj_id,
            'type': ActionType.RemoveKeep,
        }

    def gen_stop_move(self, obj_id, target, candidate):
        return {
            'actor': self.seat,
            'obj_id': obj_id,
            'type': ActionType.StopMove,
        }

    def gen_WeaponLock(self, obj_id, target, candidate):
        return {
            'actor': self.seat,
            "obj_id": obj_id,
            "type": ActionType.WeaponLock
        }

    def gen_WeaponUnFold(self, obj_id, target, candidate):
        return {
            'actor': self.seat,
            "obj_id": obj_id,
            "type": ActionType.WeaponUnFold
        }

    def gen_cancel_JM_plan(self, obj_id, target, candidate):
        return {
            'actor': self.seat,
            "obj_id": obj_id,
            "type": ActionType.CancelJMPlan
        }

    def get_city(self):
        cities = []
        for city in self.observation['cities']:
            if city['flag'] != self.color:
                cities.append(city['coord'])
        return random.choice(cities)


# Mix
class Agent(AgentBase):
    def __init__(self, scenario, model):
        super(Agent, self).__init__(scenario, model)
        self.model = model

    def get_action(self, action, obs):
        total_actions = []
        valid_actions = self.observation['valid_actions']

        obj_id = self.observation['operators'][action['select_unit'].item()]['obj_id']
        action_type = action['action_type'].item()
        location = action['location'].item()
        location = location % HP.embed_y + location // HP.embed_y * 100
        target = action['target'].item()
        if len(obs['operators']) > target:
            target = obs['operators'][target]['obj_id']

        valid_action = valid_actions[obj_id]
        if action_type in [ActionType.JMPlan, ActionType.Move]:
            target = location
        if action_type == ActionType.Occupy:
            target = self.get_city()
            total_actions.append(self.gen_move(obj_id, target, None))
        elif action_type == ActionType.Focus:
            gen_action = self.gen_action[ActionType.Shoot]
            base_action = gen_action(obj_id, target, valid_action[ActionType.Shoot])
            total_actions.append(base_action)

            target = base_action['target_obj_id']
            for coop_unit_id, coop_valid_action in obs['valid_actions'].items():
                if ActionType.Shoot in coop_valid_action:
                    valid_shoot = coop_valid_action[ActionType.Shoot]
                    for candidate in valid_shoot:
                        if candidate['target_obj_id'] == base_action['target_obj_id']:
                            total_actions.append(gen_action(coop_unit_id, candidate, valid_shoot))
                            break
        elif action_type == ActionType.Ambush:
            total_actions.append(self.gen_move(obj_id, location, None))
            self.model.add_mask_target(obj_id, location)
        elif action_type != ActionType.Pass:
            gen_action = self.gen_action[action_type]
            total_actions.append(gen_action(obj_id, target, valid_action[action_type]))

        if ActionType.Occupy in valid_action:
            total_actions.append(self.gen_occupy(obj_id, target, valid_action[ActionType.Occupy]))

        return total_actions

    def step(self, observation):
        self.observation = observation
        prep = self.model.preprocess(observation)
        _, _, all_masks = prep
        if (~all_masks[0]).all():
            return [], None, None, None, None
        action, logits, value,all_masks = self.model(prep)
        total_actions = self.get_action(action, observation)
        prep[2]=all_masks
        return total_actions, action, logits, value, prep


if __name__ == '__main__':
    a = Agent()
