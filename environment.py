import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
from utils import get_feature_adjacency_loads, example_net

class GymEnv(gym.Env):
    def __init__(self, net:pp.pandapowerNet, loads_path:str) -> None:
        super(GymEnv, self).__init__()
        np.random.seed(42)

        self.net = net
        self.net.asymmetric_load.sort_values(by='bus', inplace=True)
        self.net.asymmetric_load.reset_index(drop=True, inplace=True)

        #self.mask = np.random.choice([0, 1], size=(len(net.asymmetric_load),), p=[1./3, 2./3])
        self.mask = np.ones(len(net.asymmetric_load))
        self.action_space = spaces.Discrete(3*len(net.asymmetric_load), seed=42)
        self.observation_space = spaces.MultiBinary((len(net.asymmetric_load), 3), seed=42)
        self.init_feature, self.init_adj = get_feature_adjacency_loads(net, mask=self.mask)
        self.init_state = np.array(self.init_feature[:, :3], dtype=np.int8)
        self.state = np.array(self.init_feature[:, :3], dtype=np.int8)
        self.total_cost = 0
        self.alpha = 0.1
        self.loads = pd.read_csv(loads_path)

        assert self.loads.shape[1] == len(net.asymmetric_load)

        self.Bm = 0
        self.B_obj = 15
        self.step_count = 0
        self.max_steps = 100
        self.terminal = False
        self.trunc = False

    def _generate_obs(self, seed):
        if seed is not None:
            np.random.seed(seed)
        obs = np.zeros((len(self.net.asymmetric_load), 3), dtype=np.int8)
        for i in range(len(self.net.asymmetric_load)):
            obs[i, np.random.choice(3)] = 1
        final_obs = obs * self.mask[:, np.newaxis]
        final = np.array(final_obs, dtype=np.int8)
        return final

    def reset(self, seed=None):
        state = self._generate_obs(seed)
        self.state = state
        self.total_cost = 0
        self.terminal = False
        self.trunc = False
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        dic_act = divmod(action, 3)
        new_state = self.state
        new_state[dic_act[0], :] = np.zeros(3)
        new_state[dic_act[0], dic_act[1]] = 1
        self.total_cost = self.cost(new_state)

        reward = self.reward(new_state)

        self.state = new_state

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.trunc = True

        return self.state, reward, self.terminal, self.trunc, {}

    def reward(self, state):

        self.Bm = 0
        miss_pf = 0

        for time in range(self.loads.shape[0]):
            for i in range(self.net.asymmetric_load.shape[0]):

                if sum(state[i]) > 0:

                    if state[i, 0] == 1:

                        self.net['asymmetric_load'].loc[i, 'p_a_mw'] = self.loads.iloc[time, i]
                        self.net['asymmetric_load'].loc[i, 'p_b_mw'] = 0
                        self.net['asymmetric_load'].loc[i, 'p_c_mw'] = 0

                    elif state[i, 1] == 1:

                        self.net['asymmetric_load'].loc[i, 'p_a_mw'] = 0
                        self.net['asymmetric_load'].loc[i, 'p_b_mw'] = self.loads.iloc[time, i]
                        self.net['asymmetric_load'].loc[i, 'p_c_mw'] = 0

                    elif state[i, 2] == 1:

                        self.net['asymmetric_load'].loc[i, 'p_a_mw'] = 0
                        self.net['asymmetric_load'].loc[i, 'p_b_mw'] = 0
                        self.net['asymmetric_load'].loc[i, 'p_c_mw'] = self.loads.iloc[time, i]

            try:
                pp.runpp_3ph(self.net, max_iteration=100)
                i_a = self.net.res_trafo_3ph.i_a_lv_ka.values[0]
                i_b = self.net.res_trafo_3ph.i_b_lv_ka.values[0]
                i_c = self.net.res_trafo_3ph.i_c_lv_ka.values[0]

                max_i = max(i_a, i_b, i_c)
                mean_i = (i_a + i_b + i_c) / 3

                b_t = ((max_i - mean_i) / mean_i) * 100
                self.Bm += b_t**2

            except:
                miss_pf += 1
                continue

        if miss_pf/self.loads.shape[0] < 0.2:
            self.Bm = np.sqrt(self.Bm / self.loads.shape[0])
        else:
            #print(f"PF not converged {miss_pf} times")
            self.Bm = 1000
        rw = -self.Bm

        if self.Bm <= self.B_obj:
            self.terminal = True

        return rw

    def cost(self, state):
        cost = 0
        for i in range(state.shape[0]):
            if (state[i] - self.init_state[i]).any() != 0:
                cost += 1
        return cost

    def render(self):

        print("Current split: ",self.state.sum(axis=0))
        print("Current imbalance: ", self.Bm)
        print("\n")

    def close(self):
        pass