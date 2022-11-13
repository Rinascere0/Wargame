import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
from arch.core import Core
from arch.entity_encoder import Entity_encoder
from arch.spatial_encoder import Spatial_encoder
from arch.action_type_head import ActionTypeHead
from arch.location_head import LocationHead
from arch.target_head import TargetHead
from arch.repeat_head import RepeatHead

from lib.functions import *
from torch.distributions import Categorical
import pickle
from lib.const import *


class Model(nn.Module):
    def __init__(self, device='cpu', debug=False):
        super(Model, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(HP.base_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.core = Core()
        self.entity_encoder = Entity_encoder()
        self.spatial_encoder = Spatial_encoder()

        self.action_type_head = ActionTypeHead(debug=debug)
        self.location_head = LocationHead(debug=debug)
        self.target_head = TargetHead(debug=debug)
        self.repeat_head = RepeatHead(debug=debug)

        self.device = device
        self.debug = debug
        self.softmax = nn.Softmax(dim=1)
        self.color = 1
        self.max_entity = HP.max_entity

        self.unit_ids = []
        self.unit_mask = []
        self.unit_func = {}
        self.unit_func_rev = {}

    def init_params(self, chess_info):
        i = 0
        for j, unit in enumerate(chess_info):
            if unit['实体算子'] == 1 and unit['自有属性']['军兵种'] is not None:
                unit_id = unit['算子ID']
                self.unit_ids.append(unit_id)
                if unit['归属'] == self.color:
                    self.unit_mask.append(True)
                else:
                    self.unit_mask.append(False)
                self.unit_func[unit_id] = i
                self.unit_func_rev[i] = unit_id
                i = i + 1
        print('unit_mask', self.unit_mask)

    def gen_mask(self, chess_info):
        unit_locations = [None for _ in range(self.max_entity)]
        unit_types = [None for _ in range(self.max_entity)]
        dload_masks = [torch.zeros([self.max_entity]) for _ in range(self.max_entity)]
        load_masks = [torch.zeros([self.max_entity]) for _ in range(self.max_entity)]
        fire_masks = [torch.zeros([self.max_entity]) for _ in range(self.max_entity)]
        location_scales = [None for _ in range(self.max_entity)]
        #    action_type_masks = [None for _ in range(self.max_entity)]
        action_type_masks = [torch.tensor([[0, 0, 0, 0, 0, 0, 1]]).float() for _ in range(self.max_entity)]
        my_active_units = []
        alive_units = []

        load_id = [[] for _ in range(self.max_entity)]
        load_id_list = []
        unit_idx_list = []

        for i, unit in enumerate(chess_info):
            if unit['实体算子'] == 1 and unit['自有属性']['军兵种'] is not None and unit['状态']['歼灭'] == 0:
                unit_id = unit['算子ID']
                idx = self.unit_func[unit_id]
                unit_locations[idx] = (unit['GausX'], unit['GausY'])
                unit_types[idx] = unit['自有属性']['军兵种']
                unit_idx_list.append(i)
                alive_units.append(idx)

                for i in range(1, 9):
                    if unit['装载']['LoadID' + str(i)] is None:
                        break
                    else:
                        load_unit = unit['装载']['LoadID' + str(i)]
                        load_id[idx].append(load_unit)
                        load_id_list.append(load_unit)

        print('alive', alive_units)
        print('rev', self.unit_func_rev)
        for j in unit_idx_list:
            unit = chess_info[j]
            unit_id = unit['算子ID']
            if unit_id in load_id_list:
                continue
            if unit['归属'] == self.color and unit['状态']['歼灭'] == 0:
                my_active_units.append(unit_id)
                idx = self.unit_func[unit_id]
                my_location = (unit['GausX'], unit['GausY'])

                base_mask = torch.zeros(HP.action_num)
                base_mask[-1] = True
                for acttype in eval(unit['ActType']):
                    base_mask[ActTypeCode[acttype]] = True

                # Move Mask
                if unit['状态']['已欠机动点值'] > 0:
                    base_mask[ActType.Move] = False

                # Dload Mask

                if base_mask[ActType.DLoad]:
                    if len(load_id[idx]) > 0:
                        dload_mask = torch.zeros([self.max_entity])
                        for load_unit in load_id[idx]:
                            dload_mask[self.unit_func[load_unit]] = True
                        dload_masks[idx] = dload_mask
                    else:
                        base_mask[ActType.DLoad] = False

                # Load Mask
                if base_mask[ActType.Load]:
                    load_mask = torch.zeros(HP.max_entity)
                    if unit['装载']['LoadID4'] is None:
                        for i, location in enumerate(unit_locations):
                            if location is not None and i in alive_units and self.unit_mask[i] is True and \
                                    self.unit_ids[i] != unit_id and location == my_location and self.unit_ids[
                                i] not in load_id_list:
                                load_mask[i] = True
                        if torch.sum(load_mask) == 0:
                            base_mask[ActType.Load] = False
                        load_masks[idx] = load_mask

                # Fire Mask
                fire_mask = None

                if base_mask[ActType.Fire]:
                    fire_mask = torch.zeros(HP.max_entity)
                    weapons = unit["挂载/负荷"]
                    if unit['Vis'] is not None:
                        vis = eval(unit['Vis'])
                        #        print(unit_id,vis)
                        for i, (location, unit_type) in enumerate(zip(unit_locations, unit_types)):
                            if location is None or self.unit_func_rev[i] not in vis or i not in alive_units:
                                continue
                            #             print(self.unit_func_rev[i])
                            if self.unit_mask[i] is False:
                                dist = get_dist(my_location, location)
                                for weapon_id in range(1, 5):
                                    weapon = weapons['武器' + str(weapon_id)]
                                    if weapon is not None:
                                        if unit_type == 1 and weapon['反步兵射程'] >= dist or unit_type > 1 and weapon[
                                            '反装甲射程'] >= dist:
                                            for bullet_id in range(1, 5):
                                                bullet = weapon['弹药' + str(bullet_id)]
                                                if bullet is not None and bullet['弹药发数'] > 0 and bullet[
                                                    '所属军兵种'] == '直瞄武器弹' and (
                                                        unit_type == 1 and bullet['反步兵战斗力值（GM）'] or unit_type > 1 and
                                                        bullet['反装甲战斗力值（AP）']):
                                                    fire_mask[i] = True
                                                    break

                    if torch.sum(fire_mask) == 0:
                        base_mask[ActType.Fire] = False
                    fire_masks[idx] = fire_mask

                # Art Mask
                location_scale = [unit['GausX'], unit['GausY'], 1000000, 0]
                if base_mask[ActType.Artillery]:
                    base_mask[ActType.Artillery] = False
                    #   print('uid',unit_id)
                    for weapon_id in range(1, 5):
                        weapon = weapons['武器' + str(weapon_id)]
                        if weapon is not None:
                            for bullet_id in range(1, 5):
                                bullet = weapon['弹药' + str(bullet_id)]
                                if bullet is not None:
                                    #  print('弹药发数',bullet['弹药发数'])
                                    if bullet['所属军兵种'] == '间瞄炮弹' and bullet['弹药发数'] > 0 and (
                                            bullet['间瞄反装甲攻击值（GM/A）'] > 0 or bullet['间瞄反非装甲攻击值（GM/U）'] > 0):
                                        base_mask[ActType.Artillery] = True
                                        location_scale[2] = min(location_scale[2], weapon['间瞄近射程'])
                                        location_scale[3] = max(location_scale[3], weapon['间瞄远射程'])
                    location_scales[idx] = location_scale

            #    base_mask[ActType.Move]=False
            #    base_mask[ActType.Fire]=False
            #    base_mask[ActType.DLoad]=False
            #    base_mask[ActType.Load]=False
            #    base_mask[ActType.Scout]=False
            #    if base_mask[ActType.Artillery]:
            #        base_mask[ActType.Pass]=False

            action_type_masks[idx] = base_mask.unsqueeze(0)

        return action_type_masks, dload_masks, load_masks, fire_masks, location_scales, my_active_units

    def gen_local_state(self, state):
        state = state.expand([20, state.shape[-1]])
        #  id_one_hot = torch.zeros([20, 32])
        id_one_hot = torch.ones([20, 32]) - 1
        for i in range(20):
            id_one_hot[i][i] = 1

        state = torch.cat([state, id_one_hot], dim=1)
        return state

    def forward(self, map_info, state_info, chess_info, color):
        entity_embedding, embedded_entity, entity_prep = self.entity_encoder(chess_info)
        embedded_spatial, map_skip, map_prep = self.spatial_encoder(map_info, state_info, color)

        state = torch.cat([embedded_entity, embedded_spatial], dim=1)
        state = self.core(state)
        if self.debug:
            print('state shape:', state.shape)

        masks = self.gen_mask(chess_info)
        action_type_masks, dload_masks, load_masks, fire_masks, location_scale, my_active_units = masks

        total_actions = []
        total_logits = []
        autoregressive_embedding = None

        #     for i, mask in enumerate(dload_masks):
        #         if torch.sum(mask) != 0:
        #             print(self.unit_func_rev[i],mask)

        for unit_id in my_active_units:
            idx = self.unit_func[unit_id]
            action_type, action_type_logits, autoregressive_embedding = self.action_type_head(autoregressive_embedding,
                                                                                              state,
                                                                                              action_type_masks[idx])
            location, location_logits, autoregressive_embedding = self.location_head(autoregressive_embedding, map_skip,
                                                                                     location_scale[idx], action_type)
            target, target_logits, autoregressive_embedding = self.target_head(autoregressive_embedding,
                                                                               entity_embedding, action_type,
                                                                               load_masks[idx], dload_masks[idx],
                                                                               fire_masks[idx])
            repeat, repeat_logits, autoregressive_embedding = self.repeat_head(autoregressive_embedding, action_type)

            total_actions.append(
                {'unit_id': unit_id, 'action_type': action_type, 'location': location, 'target': target
                    , 'repeat': repeat})
            # total_logits.append([action_type_logits, location_logits, target_logits, repeat_logits])
            total_logits.append(
                torch.mean(torch.tensor([action_type_logits, location_logits, target_logits, repeat_logits])))
            with open('logits.txt', 'a') as f:
                f.write('\n')
        total_logits = torch.sum(torch.stack(total_logits))
        return total_actions, total_logits, self.unit_func_rev, (entity_prep, map_prep, masks)

    def evaluate(self, state, actions):
        entity_prep, map_prep, masks = state
        entity_embedding, embedded_entity, _ = self.entity_encoder(entity_prep, detach=True)
        embedded_spatial, map_skip, _ = self.spatial_encoder(map_prep, None, None, detach=True)

        state = torch.cat([embedded_entity, embedded_spatial], dim=1)
        state = self.core(state)
        if self.debug:
            print('state shape:', state.shape)

        value = self.critic(state)

        action_type_masks, dload_masks, load_masks, fire_masks, location_scale, my_active_units = masks

        total_logits = []
        total_entropy = []
        autoregressive_embedding = None

        for unit_id, action in zip(my_active_units, actions):
            idx = self.unit_func[unit_id]
            action_type, location, target, repeat = action['action_type'].detach(), action['location'].detach(), action[
                'target'].detach(), \
                                                    action['repeat'].detach()
            action_type_entropy, action_type_logits, autoregressive_embedding = self.action_type_head.evaluate(
                autoregressive_embedding,
                state,
                action_type_masks[idx],
                action_type)
            location_entropy, location_logits, autoregressive_embedding = self.location_head.evaluate(
                autoregressive_embedding,
                map_skip,
                location_scale[idx],
                action_type, location)
            target_entropy, target_logits, autoregressive_embedding = self.target_head.evaluate(
                autoregressive_embedding,
                entity_embedding, action_type,
                load_masks[idx],
                dload_masks[idx],
                fire_masks[idx], target)
            repeat_entropy, repeat_logits, autoregressive_embedding = self.repeat_head.evaluate(
                autoregressive_embedding,
                action_type, repeat)

            #   total_logits.append([action_type_logits, location_logits, target_logits, repeat_logits])
            #   total_entropy.append([action_type_entropy, location_entropy, target_entropy, repeat_entropy])
            total_logits.append(
                torch.mean(torch.stack([action_type_logits, location_logits, target_logits, repeat_logits])))
            total_entropy.append(
                torch.mean(torch.stack([action_type_entropy, location_entropy, target_entropy, repeat_entropy])))

        total_logits = torch.sum(torch.stack(total_logits))
        total_entropy = torch.sum(torch.stack(total_entropy))
        return value, total_logits, total_entropy


if __name__ == '__main__':
    path = '../info/'
    with open(path + 'chess.pkl', 'rb') as f:
        chessinfo = pickle.load(f)['Data']
    with open(path + 'map.pkl', 'rb') as f:
        mapinfo = pickle.load(f)
    with open(path + 'round.pkl', 'rb') as f:
        stateinfo = pickle.load(f)['Data']['当前态势']

    model = Model(debug=False)
    model.init_params(chessinfo)
    total_actions, total_logits_old, unit_func_rev = model.forward(mapinfo, stateinfo, chessinfo, 1)
    for action in total_actions:
        print(action)

    value, total_entropy, total_logits = model.evaluate(mapinfo, stateinfo, chessinfo, total_actions, 1)

    print(total_logits_old)
    print(total_logits)
    print(total_entropy)
