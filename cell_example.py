from environment import Environment

from itertools import count
from multiprocessing import Process, Lock

import time
import random
import os, sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np

class cell(Environment):
    def __init__(self, size, init_conc=0.02, target_conc=0.1, agent_range = 2, num_actions = 3, lock = None,
                 max_iteration = 5000, name = None):

        super(cell, self).__init__(size, init_conc=init_conc, target_conc=target_conc, agent_range = agent_range, 
                                   num_actions = num_actions, lock = lock, name = name, max_iteration = max_iteration)

    # def default(self, agent):
    #     curr = self.get_agent_state(agent)
    #     prev = agent.get_state()
    #     default = (agent.get_type() * (curr - prev))
    #     sames = default[default > 0.1].sum()
    #     diffs = self.alpha * default[default < -0.1].sum()
    #     return sames + diffs

def play(map, episodes, iterations, eps=1e-2):
    # map.configure(prey_reward, stuck_penalty, agent_max_age)
    # agents = map.get_agents()
    #directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (0, 0)]
    times = 0
    for episode in range(episodes):
        #print(f'episode{episode}')
        c = 0
        map.reset()
        map.save_initial()
        agents = map.get_agents()
        old_global_conc = map.get_global_conc()
        for t in count():
            t_start = time.time()
            #state = map.get_map()
            random.shuffle(agents)

            keys = [agent.get_id() for agent in agents]
            rews = {key: 0 for key in keys}
            #counts = {key: 0 for key in keys}
            #print(keys)
            #print('iteration')
            for agent in agents:
                # towards = None
                # name = map.vals_to_names[agent.get_type()]
                if agent.is_alive():
                    #print('id', agent.get_id())
                    #print('init_age', agent.get_age())
                    map.move(agent)
                    agent_state = agent.get_state()
                    action_id = agent.decide(agent_state)
                    #print('action_id', action_id)
                    #print('final_age', agent.get_age())
                    #towards = directions[action]
                rew = map.step(agent, action_id)
                rews[agent.get_id()] = rew
                #counts[name] += 1

            #print('update')
            map.update()
            #print('update', [agent.get_id() for agent in agents])

            map.record(rews)

            #next_state = map.get_map()

            time_elapsed = time.time() - t_start
            times += time_elapsed
            avg_time = times / (t + 1)
            print("I: %d\tTime Elapsed: %.2f" % (t+1, avg_time), end='\r')
            # if abs(next_state - state).sum() < eps:
            #     c += 1
            new_global_conc = map.get_global_conc()
            #print('old',old_global_conc)
            #print('new', new_global_conc)
            if abs(new_global_conc - map.target_conc) < eps and abs(new_global_conc - old_global_conc) < eps:
                c += 1
            else:
                c = 0

            if t == (iterations - 1) or c==15 or len(agents)==0:
                break

            #state = next_state
            old_global_conc = new_global_conc
        map.save(episode)
    np.save(f"{map.name}/map.npy", np.array(map.history))
    map.A_mind.save(map.name)
    #torch.save(map.A_mind, f'{map.name}/trained_model')
    print("SIMULATION IS FINISHED.")

if __name__ == '__main__':
    [_, name, iterations, init_conc, target_conc, agent_range] = sys.argv

    np.random.seed(38)
    random.seed(6)
    torch.manual_seed(15)

    episodes = 1
    iterations = int(iterations)
    l = Lock()

    args = ["Name",
            "Initial Concentration",
            "Target Concentration",
            "Agent Field of View"]

    society = cell

    play(society((50, 50), init_conc=float(init_conc), target_conc=float(target_conc), agent_range = int(agent_range), 
                 name=name, max_iteration = int(iterations), lock=l), episodes, iterations)
