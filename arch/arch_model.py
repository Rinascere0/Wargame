import json

import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
from arch.core import Core
from arch.entity_encoder import Entity_encoder
from arch.spatial_encoder import Spatial_encoder
from arch.scalar_encoder import Scalar_encoder
from arch.action_type_head import ActionTypeHead
from arch.location_head import LocationHead
from arch.select_unit_head import SelectUnitHead
from arch.select_target_head import SelectTargetHead

from lib.functions import *
from torch.distributions import Categorical
import pickle
from lib.const import *

debug = False


class ArchModel(nn.Module):
    def __init__(self, color, device='cpu'):
        super(ArchModel, self).__init__()

        self.core = Core()
        self.entity_encoder = Entity_encoder(color=color, device=device)
        self.spatial_encoder = Spatial_encoder(color=color, device=device)
        self.scalar_encoder = Scalar_encoder(device=device)
        self.action_type_head = ActionTypeHead(device=device)
        self.select_unit_head = SelectUnitHead(device=device)
        self.select_target_head = SelectTargetHead(device=device)
        self.location_head = LocationHead(device=device)
        self.critic = nn.Sequential(
            nn.Linear(HP.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.device = device
        self.debug = debug
        self.softmax = nn.Softmax(dim=1)
        self.color = color
        self.max_entity = HP.max_entity

        self.unit_ids = []

        self.time = 0
        self.dynamic_mask = {}
        self.dynamic_time = {}
        self.ambush_target = {}

    def init_info(self, map_info):
        self.map = map_info
        self.spatial_encoder.init_info(map_info.get_map_data())

    def init_mask(self, state):
        for obj in state[self.color]['operators'] + state[self.color]['passengers']:
            if obj['color'] == self.color:
                self.dynamic_mask[obj['obj_id']] = torch.ones(HP.action_num)
                self.dynamic_time[obj['obj_id']] = -1
                self.ambush_target[obj['obj_id']] = -1

    def add_mask_target(self, unit_id, target_pos):
        self.ambush_target[unit_id] = target_pos

    def add_mask(self, unit_id, unit_pos):
        target_pos = self.ambush_target[unit_id]
        if target_pos == unit_pos and self.dynamic_time[unit_id] == -1:
            self.dynamic_time[unit_id] = self.time
            self.dynamic_mask[unit_id][ActionType.Move] = 0
            self.dynamic_mask[unit_id][ActionType.Occupy] = 0

    def remove_mask(self, unit_id):
        if self.dynamic_time[unit_id] != -1:
            if self.time - self.dynamic_time[unit_id] >= 150:
                self.dynamic_time[unit_id] = -1
                self.dynamic_mask[unit_id][ActionType.Move] = 1
                self.dynamic_mask[unit_id][ActionType.Occupy] = 1

    def gen_unit_func(self, obs):
        unit_func = {}
        unit_func_rev = {}
        i = 0
        all_units = obs['operators'] + obs['passengers']
        for j, unit in enumerate(all_units):
            unit_id = unit['obj_id']
            self.unit_ids.append(unit_id)
            unit_func[unit_id] = i
            unit_func_rev[i] = unit_id
            i = i + 1
        return unit_func, unit_func_rev

    def get_unit(self, unit_id, obs):
        for bop in obs['operators']:
            if unit_id == bop['obj_id']:
                return bop

    def can_occupy(self, obs):
        for city in obs['cities']:
            if city['flag'] != self.color:
                return True
        return False

    def can_stop_move(self, unit_id, obs):
        location = self.get_unit(unit_id, obs)['cur_hex']
        max_dist, min_dist = 0, 40
        for city in obs["cities"]:
            dist = self.map.get_distance(location, city['coord'])
            max_dist = max(max_dist, dist)
            min_dist = min(min_dist, dist)
        return min_dist >= 20 or max_dist <= 10

    def gen_mask(self, obs):
        action_type_masks = []
        select_target_masks = []
        enemy_pos = get_enemy_pos(obs, self.color)
        city_pos = get_city_pos(obs, self.color)

        unit_map = {}
        for idx, unit in enumerate(obs['operators']):
            unit_map[unit['obj_id']] = idx

        for unit in obs['operators']:
            if unit['color'] != self.color:
                continue
            action_type_mask = torch.zeros(HP.action_num)
            select_target_mask = torch.zeros(1, HP.max_entity)

            unit_id = unit['obj_id']
            self.add_mask(unit_id, unit['cur_hex'])
            if unit_id in obs['valid_actions']:
                valid_action = obs['valid_actions'][unit_id]
                action_type_mask[ActionType.Pass] = 1

                for action_type in valid_action:
                    targets = valid_action[action_type]
                    action_type_mask[action_type] = 1

                if ActionType.Move in valid_action:
                    if self.can_occupy(obs):
                        action_type_mask[ActionType.Occupy] = 1

                if ActionType.Shoot in valid_action:
                    for candidate in valid_action[ActionType.Shoot]:
                        select_target_mask[0][unit_map[candidate['target_obj_id']]] = 1

                action_type_mask[ActionType.ChangeState] = 0
                action_type_mask[ActionType.WeaponLock] = 0
                action_type_mask[ActionType.WeaponUnFold] = 0

                #     if action_type_mask[ActionType.Shoot]:
                #         action_type_mask[ActionType.Focus]=True

                #              if action_type_mask[ActionType.Move]:
                #                  action_type_mask[ActionType.Ambush]=True

                if action_type_mask[ActionType.StopMove]:
                    action_type_mask[ActionType.StopMove] = 0
                    for target_pos in enemy_pos:
                        dist = self.map.get_distance(target_pos, unit['cur_hex'])
                        if unit['type'] == 1 and dist <= 4 or unit['type'] == 2 and dist <= 8:
                            action_type_mask[ActionType.StopMove] = 1

                    for target_pos in city_pos:
                        if self.map.get_distance(target_pos, unit['cur_hex']) < 5:
                            action_type_mask[ActionType.StopMove] = 1

            self.remove_mask(unit_id)
            action_type_mask *= self.dynamic_mask[unit_id]
            action_type_masks.append(action_type_mask)
            select_target_masks.append(select_target_mask)

        select_unit_mask = torch.sum(torch.stack(action_type_masks), dim=1) != 0
        vacant_num = HP.max_entity - select_unit_mask.shape[0]
        vacant_unit_mask = torch.zeros([1, vacant_num]).bool()
        select_unit_mask = torch.cat([select_unit_mask.unsqueeze(0), vacant_unit_mask], dim=1)

        if debug:
            print('select_unit_mask', select_unit_mask)

        return select_unit_mask, action_type_masks, select_target_masks

    def preprocess(self, state):
        self.time = state['time']['cur_step']
        entity_prep = self.entity_encoder.preprocess(state)
        map_prep = self.spatial_encoder.pre_process(state)
        #   scalar_prep = self.scalar_encoder.preprocess(state)
        all_masks = self.gen_mask(state)
        #    return entity_prep, map_prep, scalar_prep, all_masks
        return [entity_prep, map_prep, all_masks]

    def forward(self, prep):
        with torch.no_grad():
            entity_prep, spatial_prep, all_masks = prep
            select_unit_mask, action_type_mask, select_target_mask = all_masks

            entity_embedding, embedded_entity, unit_num = self.entity_encoder(entity_prep)
            map_skip, embedded_spatial = self.spatial_encoder(spatial_prep)

            state = torch.cat([embedded_entity, embedded_spatial], dim=1)
            state = self.core(state)

            state_value = self.critic(state).squeeze(1)

            if debug:
                print('state shape:', state.shape)

            batch_size = state.shape[0]

            # NOTE:action type is valid if any unit can use it
            action_type_mask = torch.sum(action_type_mask, dim=0) > 0

            # first select action type
            action_type, action_type_logits, autoregressive_embedding = self.action_type_head(state,
                                                                                              all_action_type_mask)

            # NOTE:target is valid if any unit can select with action
            select_target_mask = torch.sum(select_target_mask[:, action_type]) > 0

            # then select target
            select_target, select_target_logits, autoregressive_embedding = self.select_target_head(
                autoregressive_embedding, entity_embedding, action_type, select_target_mask)

            # NOTE:unit is valid if it can select current action and target
            select_unit_mask = select_target_mask[:, action_type, select_target]

            # then select units
            select_unit, select_unit_logits, autoregressive_embedding = self.select_unit_head(autoregressive_embedding,
                                                                                              entity_embedding,
                                                                                              select_unit_mask,
                                                                                              unit_num)
            # finally select location
            location, location_logits, autoregressive_embedding = self.location_head(autoregressive_embedding,
                                                                                     map_skip,
                                                                                     action_type)

            if debug:
                print('location', location)
                print('location_logits', location_logits.shape)

            actions = {'action_type': action_type.detach().cpu(),
                       'select_unit': select_unit.detach().cpu(),
                       'location': location.detach().cpu(),
                       'target': select_target.detach().cpu()
                       }

            logits = (
                action_type_logits.detach().cpu(), select_unit_logits.detach().cpu(), location_logits.detach().cpu(),
                select_target_logits.detach().cpu())
            if debug:
                print('actions', actions)
                print('logits', logits)

        return actions, logits, state_value, (action_type_mask, select_target_mask, select_unit_mask)

    def evaluate(self, prep, actions):
        entity_prep, spatial_prep, all_masks = prep
        select_unit_mask, action_type_mask, select_target_mask = all_masks
        action_type, select_unit, location, select_target = actions

        entity_embedding, embedded_entity, unit_num = self.entity_encoder(entity_prep)
        map_skip, embedded_spatial = self.spatial_encoder(spatial_prep)

        select_unit, action_type, location, select_target = actions

        state = torch.cat([embedded_entity, embedded_spatial], dim=1)
        state = self.core(state)

        state_value = self.critic(state).squeeze(1)

        batch_size = state.shape[0]

        # NOTE:action type is valid if any unit can use it
        action_type_mask = torch.sum(action_type_mask, dim=0) > 0

        # first select action type
        action_type_entropy, action_type_logits, autoregressive_embedding = self.action_type_head.evaluate(state,
                                                                                                           action_type_mask,
                                                                                                           action_type)

        # NOTE:target is valid if any unit can select with action
        select_target_mask = torch.sum(select_target_mask[:, action_type]) > 0

        # then select target
        select_target_entropy, select_target_logits, autoregressive_embedding = self.select_target_head(
            autoregressive_embedding, entity_embedding, action_type, select_target_mask)

        # NOTE:unit is valid if it can select current action and target
        select_unit_mask = select_target_mask[:, action_type, select_target]

        # then select units
        select_unit_entropy, select_unit_logits, autoregressive_embedding = self.select_unit_head(
            autoregressive_embedding,
            entity_embedding,
            select_unit_mask,
            unit_num)

        # finally select location
        location_entropy, location_logits, autoregressive_embedding = self.location_head(autoregressive_embedding,
                                                                                         map_skip,
                                                                                         action_type)

        logits = (select_unit_logits, action_type_logits, location_logits, select_target_logits)
        entropys = torch.cat(
            [select_unit_entropy.unsqueeze(1), action_type_entropy.unsqueeze(1), location_entropy.unsqueeze(1),
             select_target_entropy.unsqueeze(1)], dim=-1)
        entropys = torch.sum(entropys, dim=1)

        if debug:
            print('entropy', entropys)
            print('logits', logits)

        return logits, entropys, state_value


def get_action_batch(actions):
    select_units = []
    action_types = []
    locations = []
    for action in actions:
        select_unit, action_type, location = action['select_unit'], action['action_type'], action['location']
        select_units.append(select_unit)
        action_types.append(action_type)
        locations.append(location)
    select_units = torch.stack(select_units, dim=0).squeeze(1)
    action_types = torch.stack(action_types, dim=0).squeeze(1)
    locations = torch.stack(locations, dim=0).squeeze(1)
    print("select_units", select_units.shape)
    print('action_types', action_types.shape)
    print('locations', locations.shape)
    return select_units, action_types, locations


def get_prep(preps):
    entity_preps = []
    spatial_preps = []
    #  scalar_preps = []
    select_unit_masks = []
    action_type_masks = []
    masks = []
    for prep in preps:
        entity_prep, spatial_prep, scalar_prep, mask = prep
        entity_preps.append(entity_prep)
        spatial_preps.append(spatial_prep)
        #  scalar_preps.append(scalar_prep)
        select_unit_mask, action_type_mask = mask
        select_unit_masks.append(select_unit_mask)
        action_type_masks.append(action_type_mask[0])
    entity_preps = torch.stack(entity_preps, dim=0).squeeze(1)
    spatial_preps = torch.stack(spatial_preps, dim=0).squeeze(1)
    # scalar_preps = torch.stack(scalar_preps, dim=0).squeeze(1)
    select_unit_masks = torch.stack(select_unit_masks, dim=0).squeeze(1)
    if debug:
        print('entity_prep', entity_preps.shape)
        print('spatial_prep', spatial_preps.shape)
        #    print('scalar_prep', scalar_preps.shape)
        print('select_unit_mask', select_unit_masks.shape)
    #  return entity_preps, spatial_preps, scalar_preps, (select_unit_masks, action_type_masks)
    return entity_preps, spatial_preps, (select_unit_masks, action_type_masks)


def test():
    model = ArchModel()
    with open('../lib/obs.pkl', 'rb') as f:
        obs = pickle.load(f)
    with open('../lib/map.pkl', 'rb') as f:
        map_info = pickle.load(f)
    model.init_info(None, map_info)
    preps = [model.preprocess(obs[0]), model.preprocess(obs[0])]
    prep = get_prep(preps)
    actions = []
    logits = []
    for i, p in enumerate(preps):
        print('batch', i)
        total_actions, total_logits = model.forward(p)
        actions.append(total_actions)
        logits.append(total_logits)
        print()

    print('>> get_learning_batch')
    actions = get_action_batch(actions)
    logits = torch.cat(logits)
    print()

    print('>> evaluate')
    logits, entropy = model.evaluate(prep, actions)


if __name__ == '__main__':
    test()
