import os
import gzip
import math
import copy
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

from multiprocessing import Queue, Lock

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agent import Agent
from mind import Mind

class Environment:
    def __init__(self, size, init_conc=0.02, target_conc=0.1, agent_range = 5, num_actions = 3, lock = None, 
                 name=None, max_iteration = 5000, boundary=False):
        """
        Parameters:
        size: size of map (n x n)
        init_conc: initial concentration of cells
        num_actions = number of actions (divide, die, do nothing)
        """
        (H, W) = self.size = size
        input_size = (2*agent_range + 1) ** 2  # size of observation window
        self.boundary_exists = boundary
        if lock:
            self.lock = lock
        else:
            self.lock = Lock()

        self.A_mind = Mind(input_size, num_actions, self.lock, Queue())

        self.max_iteration = max_iteration
        self.lock = lock

        self.init_conc = init_conc
        assert init_conc <= 1, 'Initial concentration of cells needs to be less than or equals to one'
        self.target_conc = target_conc
        assert target_conc <= 1, 'Target concentration of cells needs to be less than or equals to one'

        self.hzn = agent_range
        # self.vals = [0, 1]
        # self.names_to_vals = {"empty": 0, "occupied": 1}
        # self.vals_to_names = {v: k for k, v in self.names_to_vals.items()}
        # self.vals_to_index = {0: 0, 1: 1}

        self.num_grids = size[0] * size[1]

        self.id_to_lives = {}

        self.crystal = np.zeros((max_iteration, H, W, 4)) # global_conc, local_conc, age, id
        self.history = []
        self.id_track = []
        self.records = []

        self.args = [self.init_conc, self.target_conc, self.hzn]
        if name:
            self.name = name
        else:
            self.name = abs(hash(tuple([self] + self.args)))

        if not os.path.isdir(str(self.name)):
            os.mkdir(str(self.name))
            os.mkdir(str(self.name)+'/episodes')
            self.map, self.agents, self.loc_to_agent, self.id_to_agent = self._generate_map()
            self._set_initial_states()
            self.mask = self._get_mask()
            self.crystal = np.zeros((max_iteration, H, W, 4)) # global_conc, local_conc, age, id
            self.iteration = 0
        else:
            assert False, "There exists an experiment with this name."
        
        # new agents from divide
        self.new_agents = []
    
    def reset(self):
      self.map, self.agents, self.loc_to_agent, self.id_to_agent = self._generate_map()
      self._set_initial_states()
      self.mask = self._get_mask()
      #self.crystal = np.zeros((self.max_iteration, self.size[0], self.size[1], 4)) # global_conc, local_conc, age, id
      self.iteration = 0

    def configure(self, init_conc, target_conc):
        self.init_conc = init_conc
        self.target_conc = target_conc

    def get_agents(self):
        return self.agents

    def get_map(self):
        return self.map.copy()

    def move(self, agent):
        (i, j) = loc = agent.get_loc()
        # choose to not move, move up, down, left, right randomly
        direction = np.random.randint(5)
        if direction == 1:
            by = (0, 1)
        elif direction == 2:
            by = (0, -1)
        elif direction == 3:
            by = (-1, 0)
        elif direction == 4:
            by = (1, 0)
        else:
            by = (0, 0)
        (i_n, j_n) = to = self._add((i, j), by)
        #(i_n, j_n) = to = agent.get_decision()
        # if there is an agent occupying final location then do not move
        if self.map[i_n, j_n] == 0:
            self.map[i, j] = 0
            self.map[i_n, j_n] = 1
            agent.set_loc(to)
            self.loc_to_agent[to] = agent
            del self.loc_to_agent[loc]
    
    def die(self, agent):
        agent.alive = False
        (i, j) = loc = agent.get_loc()
        self.map[i, j] = 0
        del self.loc_to_agent[loc]
        # remove dead agent
        #self.agents.remove(agent)
    
    def divide(self, agent):
        (i, j) = loc = agent.get_loc()
        by_list = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        free_space = [k for k, by in enumerate(by_list) if self.map[self._add((i, j), by)]==0]
        #print('free space', free_space)
        # if there are agents occupying any of the final location then do not divide
        if len(free_space) > 0:
            divide_location = random.choice(free_space)
            #print('divide location', divide_location)
            # choose to divide above, below, left or right of original cell randomly
            # divide_location = np.random.randint(4)
            # if divide_location == 0:
            #     by = (0, 1)
            # elif divide_location == 1:
            #     by = (0, -1)
            # elif divide_location == 2:
            #     by = (-1, 0)
            # elif divide_location == 3:
            #     by = (1, 0)
            by = by_list[divide_location]
            (i_n, j_n) = to = self._add((i, j), by)
        #if self.map[i_n, j_n] == 0:
            self.map[i_n, j_n] = 1
            # add divided agent 1 (at divide location)
            new_agent_id_1 = max(self.id_to_agent.keys()) + 1
            #print('divide1',new_agent_id_1)
            new_agent_1 = Agent(new_agent_id_1, to, self.A_mind)
            self.loc_to_agent[(i_n, j_n)] = new_agent_1
            self.id_to_agent[new_agent_id_1] = new_agent_1
            state_1 = self.get_agent_state(new_agent_1)
            new_agent_1.set_current_state(state_1)
            self.new_agents.append(new_agent_1)
            # add divided agent 2 (at original location)
            new_agent_id_2 = max(self.id_to_agent.keys()) + 1
            #print('divide2',new_agent_id_2)
            new_agent_2 = Agent(new_agent_id_2, loc, self.A_mind)
            self.loc_to_agent[(i, j)] = new_agent_2
            self.id_to_agent[new_agent_id_2] = new_agent_2
            state_2 = self.get_agent_state(new_agent_2)
            new_agent_2.set_current_state(state_2)
            self.new_agents.append(new_agent_2)
            # remove original agent
            agent.divided = True
            agent.alive = False
            #self.agents.remove(agent)

    def step(self, agent, action_id):
        if agent.is_alive():
            (i, j) = agent.get_loc() # current location
            assert self.loc_to_agent[(i, j)]
            # agent takes action
            if action_id == 1:
                self.divide(agent)
                #rew = -np.abs(self.target_conc - self.get_global_conc())
                done = agent.divided  # agent might not divide if final location occupied
            elif action_id == 2:
                self.die(agent)
                #rew = -np.abs(self.target_conc - self.get_global_conc())
                done = True   
            else:
                #rew = -0.5*np.abs(self.target_conc - self.get_agent_local_conc(agent)) -0.5*np.abs(self.target_conc - self.get_global_conc())
                done = False
            rew = -np.abs(self.target_conc - self.get_global_conc())
            if self.get_global_conc() == 0 and self.target_conc != 0:
                rew -= 5
            self.update_agent(agent, rew, done)
            agent.clear_decision()
        return rew

    def update_agent(self, agent, rew, done):
        #print(agent.get_id())
        state = self.get_agent_state(agent)
        agent.set_next_state(state)
        agent.update(rew, done)
        return rew

    def update(self):
        self.iteration += 1
        self.history.append(self.map.copy())

        self.A_mind.train()
        #print(self.A_mind.get_losses())
        a_local_conc = []
        a_ages = []
        a_ids = []
        id_track = np.zeros(self.map.shape)
        self.deads = []
        self.deads_age = []
        self.divided = []
        self.divided_age = []
        self.remove_agents = []
        # add new agents from divide
        self.agents.extend(self.new_agents)
        #print('agents', [agent.get_id() for agent in self.agents])
        for agent in self.agents:
            age = agent.get_age()
            idx = agent.get_id()
            #print('idx', idx)

            if agent.is_alive():
                i, j = agent.get_loc()
                id_track[i, j] = idx
                local_conc = self.get_agent_local_conc(agent)
                global_conc = self.get_global_conc()
                self.crystal[self.iteration - 1, i, j] = [global_conc, local_conc, age, idx]
                # print('alive age', age)
                # print('alive idx', idx)

                a_local_conc.append(str(local_conc))
                a_ages.append(str(age))
                a_ids.append(str(idx))
            else:
                #print('dead_id', agent.get_id())
                if agent.divided==True:
                    self.divided.append(str(idx))
                    self.divided_age.append(str(age))
                    #print('divided', self.divided)
                else:
                    self.deads.append(str(idx))
                    self.deads_age.append(str(age))
                    #print('deads', self.deads)
                self.remove_agents.append(agent)
        #print([agent.get_id() for agent in self.remove_agents])
        for removed_agent in self.remove_agents:
            self.agents.remove(removed_agent)
        # clear new agents list
        self.new_agents.clear()

        self.id_track.append(id_track)
        a_local_conc = " ".join(a_local_conc)

        a_ages = " ".join(a_ages)

        a_ids = " ".join(a_ids)

        divided = " ".join(self.divided)

        divided_age = " ".join(self.divided_age)

        deads = " ".join(self.deads)

        deads_age = " ".join(self.deads_age)

        with open("%s/episodes/a_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (self.iteration, self.get_global_conc(), a_local_conc, a_ages, a_ids, divided, divided_age, deads, deads_age))

        if self.iteration == self.max_iteration - 1:
            A_losses = self.A_mind.get_losses()
            np.save("%s/episodes/a_loss.npy" % self.name, np.array(A_losses))
    
    def update_eval(self):
        self.iteration += 1
        self.history.append(self.map.copy())

        #self.A_mind.train()
        self.A_mind.network.eval()
        self.A_mind.target_network.eval()
        a_local_conc = []
        a_ages = []
        a_ids = []
        id_track = np.zeros(self.map.shape)
        self.deads = []
        self.deads_age = []
        self.divided = []
        self.divided_age = []
        self.remove_agents = []
        # add new agents from divide
        self.agents.extend(self.new_agents)
        #print('agents', [agent.get_id() for agent in self.agents])
        for agent in self.agents:
            age = agent.get_age()
            idx = agent.get_id()
            #print('idx', idx)

            if agent.is_alive():
                i, j = agent.get_loc()
                id_track[i, j] = idx
                local_conc = self.get_agent_local_conc(agent)
                global_conc = self.get_global_conc()
                self.crystal[self.iteration - 1, i, j] = [global_conc, local_conc, age, idx]
                # print('alive age', age)
                # print('alive idx', idx)

                a_local_conc.append(str(local_conc))
                a_ages.append(str(age))
                a_ids.append(str(idx))
            else:
                #print('dead_id', agent.get_id())
                if agent.divided==True:
                    self.divided.append(str(idx))
                    self.divided_age.append(str(age))
                    #print('divided', self.divided)
                else:
                    self.deads.append(str(idx))
                    self.deads_age.append(str(age))
                    #print('deads', self.deads)
                self.remove_agents.append(agent)
        #print([agent.get_id() for agent in self.remove_agents])
        for removed_agent in self.remove_agents:
            self.agents.remove(removed_agent)
        # clear new agents list
        self.new_agents.clear()

        self.id_track.append(id_track)
        a_local_conc = " ".join(a_local_conc)

        a_ages = " ".join(a_ages)

        a_ids = " ".join(a_ids)

        divided = " ".join(self.divided)

        divided_age = " ".join(self.divided_age)

        deads = " ".join(self.deads)

        deads_age = " ".join(self.deads_age)

        with open("%s/episodes/a_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (self.iteration, self.get_global_conc(), a_local_conc, a_ages, a_ids, divided, divided_age, deads, deads_age))

        if self.iteration == self.max_iteration - 1:
            A_losses = self.A_mind.get_losses()
            np.save("%s/episodes/a_loss.npy" % self.name, np.array(A_losses))
    
    def save_initial(self):
        self.history.append(self.map.copy())
        a_local_conc = []
        a_ages = []
        a_ids = []
        id_track = np.zeros(self.map.shape)
        self.deads = []
        self.deads_age = []
        self.divided = []
        self.divided_age = []
        for agent in self.agents:
            age = agent.get_age()
            idx = agent.get_id()

            if agent.is_alive():
                i, j = agent.get_loc()
                id_track[i, j] = idx
                local_conc = self.get_agent_local_conc(agent)
                global_conc = self.get_global_conc()
                self.crystal[self.iteration - 1, i, j] = [global_conc, local_conc, age, idx]

                a_local_conc.append(str(local_conc))
                a_ages.append(str(age))
                a_ids.append(str(idx))
            # else:
            #     if agent.divided==True:
            #         self.divided.append([agent.get_age(), agent.get_id()])
            #     else:
            #         self.deads.append([agent.get_age(), agent.get_id()])
            #     self.agents.remove(agent)

        self.id_track.append(id_track)
        a_local_conc = " ".join(a_local_conc)

        a_ages = " ".join(a_ages)

        a_ids = " ".join(a_ids)

        divided = " ".join(self.divided)

        divided_age = " ".join(self.divided_age) 

        deads = " ".join(self.deads)

        deads_age = " ".join(self.deads_age)

        with open("%s/episodes/a_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (self.iteration, self.get_global_conc(), a_local_conc, a_ages, a_ids, divided, divided_age, deads, deads_age))

    def shuffle(self):
        map = np.zeros(self.size)
        loc_to_agent = {}

        locs = [(i, j) for i in range(self.map.shape[0]) for j in range(self.map.shape[1]) if self.map[i, j] == 0]
        random.shuffle(locs)
        id_track = np.zeros(self.map.shape)
        for i, agent in enumerate(self.agents):
            loc = locs[i]
            agent.respawn(loc)
            loc_to_agent[loc] = agent
            map[loc] = 1
            id_track[loc] = agent.get_id()

        self.map, self.loc_to_agent = map, loc_to_agent
        self._set_initial_states()
        self.history = [map.copy()]
        self.id_track = [id_track]
        self.records = []
        self.iteration = 0

    def record(self, rews):
        self.records.append(rews)

    def save(self, episode):
        f = gzip.GzipFile('%s/crystal.npy.gz' % self.name, "w")
        np.save(f, self.crystal)
        f.close()

    def save_agents(self):
        self.lock.acquire()
        pickle.dump(self.agents, open("agents/agent_%s.p" % (self.name), "wb" ))
        self.lock.release()

    def get_agent_state(self, agent):
        hzn = self.hzn
        i, j = agent.get_loc()
        fov = np.zeros((2 * hzn + 1, 2 *  hzn + 1)) - 2
        if self.boundary_exists:
            start_i, end_i, start_j, end_j = 0, 2 * hzn + 1, 0, 2 * hzn + 1
            if i < hzn:
                start_i = hzn - i
            elif i + hzn - self.size[0] + 1 > 0:
                end_i = (2 * hzn + 1) - (i + hzn - self.size[0] + 1)
            if j < hzn:
                start_j = hzn - j
            elif j + hzn - self.size[1] + 1 > 0:
                end_j = (2 * hzn + 1) - (j + hzn - self.size[1] + 1)
            i_upper = min(i + hzn + 1, self.size[0])
            i_lower = max(i - hzn, 0)

            j_upper = min(j + hzn + 1, self.size[1])
            j_lower = max(j - hzn, 0)

            fov[start_i: end_i, start_j: end_j] = self.map[i_lower: i_upper, j_lower: j_upper].copy()
        else:
            for di in range(-hzn, hzn+1):
                for dj in range(-hzn, hzn+1):
                    fov[hzn + di, hzn + dj] = self.map[(i+di) % self.size[0], (j+dj) % self.size[1]]

        fov[hzn, hzn] = 1
        return fov

    def get_agent_local_conc(self, agent):
        fov = self.get_agent_state(agent)
        local_conc = fov.sum() / (fov.shape[0] * fov.shape[1])
        return local_conc
    
    def get_global_conc(self):
        global_conc = self.map.sum() / (self.map.shape[0] * self.map.shape[1])
        return global_conc

    def _to_csv(self, episode):
        with open("episodes/%s_%s.csv" % (episode, self.name), 'w') as f:
            f.write(', '.join(self.records[0].keys()) + '\n')
            proto = ", ".join(['%.3f' for _ in range(len(self.records[0]))]) + '\n'
            for rec in self.records:
                f.write(proto % tuple(rec.values()))

    def _add(self, fr, by):
        i, j = fr
        di, dj = by
        if self.boundary_exists:
            i_n = min(max(i + di, 0), self.size[0] - 1)
            j_n = min(max(j + dj, 0), self.size[1] - 1)
        else:
            i_n = (i + di) % self.size[0]
            j_n = (j + dj) % self.size[1]
        return (i_n, j_n)

    def _get_mask(self):
        mask = []
        for i, row in enumerate(self.map):
            foo = []
            for j, col in enumerate(row):
                foo.append((-1) ** (i + j))
            mask.append(foo)
        return np.array(mask)

    def _generate_map(self):
        map = np.zeros(self.size)
        loc_to_agent = {}
        id_to_agent = {}
        agents = []
        idx = 0
        num_agents = int(self.init_conc * self.size[0] * self.size[1])
        locs = [(i, j) for i in range(map.shape[0]) for j in range(map.shape[1]) if map[i, j] == 0]
        random.shuffle(locs)
        locs = locs[:num_agents]
        for loc in locs:
            (i, j) = (loc[0], loc[1])
            agent = Agent(idx, (i, j), self.A_mind)
            loc_to_agent[(i, j)] = agent
            id_to_agent[idx] = agent
            agents.append(agent)
            idx += 1
            map[i, j] = 1
        return map, agents, loc_to_agent, id_to_agent
        # for i, row in enumerate(map):
        #     for j, col in enumerate(row):
        #         val = np.random.choice(self.vals, p=self.probs)
        #         if not val == self.names_to_vals["free"]:
        #             if val == self.names_to_vals["A"]:
        #                 mind = self.A_mind
        #             elif val == self.names_to_vals["B"]:
        #                 mind = self.B_mind
        #             else:
        #                 assert False, 'Error'
        #             agent = Agent(idx, (i, j), mind)
        #             loc_to_agent[(i, j)] = agent
        #             id_to_agent[idx] = agent
        #             agents.append(agent)
        #             idx += 1
        #         map[i, j] = 1
        # return map, agents, loc_to_agent, id_to_agent

    def predefined_initialization(self, file):
        with open(file) as f:
            for i, line in enumerate(f):
                if not i:
                    keys = [key.strip() for key in line.rstrip().split(',')]
                line.rstrip().split(',')

    def _set_initial_states(self):
        for agent in self.agents:
            state = self.get_agent_state(agent)
            agent.set_current_state(state)

    # def _count(self, arr):
    #     cnt = np.zeros(len(self.vals))
    #     arr = arr.reshape(-1)
    #     for elem in arr:
    #         if elem in self.vals_to_index:
    #             cnt[self.vals_to_index[elem]] += 1
    #     return cnt
