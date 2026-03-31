
from dataclasses import dataclass
import torch
import time
from cetsp_socp import gurobi_socp
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    node_xy: torch.Tensor
    radius: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.node_xy = None
        self.radius = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        node_xy, radius = get_random_problems(batch_size, self.problem_size)
        self.node_xy = node_xy
        self.radius = radius
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.node_xy = augment_xy_data_by_8_fold(self.node_xy)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem)
        self.step_state.selected_count = self.selected_count

        reward = None
        done = False
        return Reset_State(self.node_xy, self.radius), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, test=False):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)
        self.step_state.selected_count = self.selected_count


        # returning values
        done = (self.selected_count == self.problem_size + 1)
        if test:
            if done:
                reward, loc_x, loc_y = self._get_travel_distance(test)  # note the minus sign!
                reward = -reward
                return self.step_state, reward, done, loc_x, loc_y
            else:
                reward = None
                return self.step_state, reward, done, None, None

        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self, test=False):
        # bs, ps, _ = self.selected_node_list.size()
        complete_sequence = torch.cat((self.selected_node_list, self.selected_node_list[:, :, 0][:, :, None]), -1)
        gathering_index = complete_sequence.unsqueeze(3).expand(self.batch_size, -1, self.problem_size+2, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.node_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size+1, 2)
        radius_expanded = self.radius[:, None, :].expand(self.batch_size, self.pomo_size, self.problem_size+1)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index).reshape(-1, self.problem_size+2, 2)
        ordered_radius = radius_expanded.gather(dim=2, index=complete_sequence).reshape(-1, self.problem_size+2)
        # shape: (batch, pomo, problem, 2)
        loc, radius = ordered_seq.cpu().numpy(), ordered_radius.cpu().numpy()

        if test:
            dist_list, loc_x_list, loc_y_list = [], [], []
            for i in range(self.batch_size * self.pomo_size):

                dist, x, y = gurobi_socp(loc[i], radius[i], 16, test=test)
                dist_list.append(dist)
                loc_x_list.append(x)
                loc_y_list.append(y)

            travel_distances = torch.FloatTensor(dist_list).to(gathering_index.device).reshape(self.batch_size,
                                                                                               self.pomo_size)

            return travel_distances, loc_x_list, loc_y_list

        dist_list = []
        for i in range(self.batch_size*self.pomo_size):
            # start_time = time.time()
            dist = gurobi_socp(loc[i], radius[i], 16)
            dist_list.append(dist)
            # duration = time.time() - start_time
            # print(duration)
        travel_distances = torch.FloatTensor(dist_list).to(gathering_index.device).reshape(self.batch_size, self.pomo_size)

        return travel_distances

