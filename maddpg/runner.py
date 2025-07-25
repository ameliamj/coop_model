from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from updater import Updater
from gaze import Gaze
from torch.nn import functional as F



class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns1 = []
        returns2 = []
        seeds = []
        agent_names = ['adversary_0', 'agent_0']
        gazer = Gaze(self.args.gaze_type, self.env)
        for time_step in tqdm(range(self.args.time_steps)):
            print("timestep: ", time_step)
            # Reset environment at start of each episode
            if time_step % self.episode_limit == 0:
                seed = np.random.randint(0, 1000)
                seeds.append(seed)
                print("hi")
                s, _ = self.env.reset(seed=seed)
                print("s: ", s)
                s = [s[agent_names[0]], s[agent_names[1]]]
                for i, state in enumerate(s):
                    print(f"Original obs shape for agent {i}: {state.shape}")
                    if self.args.lever_cue != 'none':
                        s[i] = np.concatenate((state, [0, 0]))
                    else:
                        s[i] = np.concatenate((state, [0]))
                    if self.args.lever_action: #NEW_CODE
                        s[i] = np.concatenate((s[i], [0]))  # lever_action #NEW_CODE
                        s[i] = np.concatenate((s[i], [0]))
                    print(f"Augmented obs shape for agent {i}: {s[i].shape}")
                        
                ha = Updater.init_hidden(hidden_size=64)
                ha_next = Updater.init_hidden(hidden_size=64)
                hc = Updater.init_hidden(hidden_size=64)
                hc_next = Updater.init_hidden(hidden_size=64)
                ca = Updater.init_hidden(hidden_size=64)
                ca_next = Updater.init_hidden(hidden_size=64)
                cc = Updater.init_hidden(hidden_size=64)
                cc_next = Updater.init_hidden(hidden_size=64)
                
                updater = Updater(self.args, self.env)
            
            # Select actions for both agents
            u = []
            actions = {}
            gaze_actions = [0, 0]
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):              
                    #print(f"Agent {agent_id} observation shape before gaze: {s[agent_id].shape}")
                    s[agent_id] = gazer.gaze(s[agent_id], gaze_actions[agent_id], agent_id)
                    #print(f"Agent {agent_id} observation shape after gaze: {s[agent_id].shape}")
                    
                    action, ha_next[agent_id], ca_next[agent_id] = agent.select_action(s[agent_id], ha[agent_id], ca[agent_id], noise_rate=self.noise, epsilon=self.epsilon)
                    print("\nAgent: ", agent_names[agent_id])
                    u.append(action)
                    actions[agent_names[agent_id]], gaze_actions[agent_id] = Updater.probs_to_actions(action, self.args.lever_action)
                    print("Action: ", actions[agent_names[agent_id]])

                for agent_id, agent in enumerate(self.agents):
                    temp_val, hc_next[agent_id], cc_next[agent_id] = agent.get_value(s[:self.args.n_agents], u, hc[agent_id], cc[agent_id], agent_id=agent_id)

            # Do action
            #print("actions: ", actions)
            temp_actions = {}
            for name in agent_names:
                temp_actions[name] = actions[name] if actions[name] != 3 else 0
                
            #print("temp_actions: ", temp_actions)
            s_next, r, done, _, info = self.env.step(temp_actions)
        
            
            s_next = [s_next[agent_names[0]], s_next[agent_names[1]]]

            # Update rewards, coord_times, and cues
            r, s_next = updater.update(s_next, time_step, actions, gaze_actions)
            # Update buffer and save results
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], ha[:self.args.n_agents], ha_next[:self.args.n_agents], hc[:self.args.n_agents], hc_next[:self.args.n_agents], ca[:self.args.n_agents], ca_next[:self.args.n_agents], cc[:self.args.n_agents], cc_next[:self.args.n_agents])
            s = s_next
            ha = ha_next
            hc = hc_next
            ca = ca_next
            cc = cc_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and (time_step + 1) % self.args.evaluate_rate == 0:
                return1, return2, gaze_actions1, gaze_actions2 = self.evaluate()
                returns1.append(return1)
                returns2.append(return2)
                plt.figure()
                plt.plot(range(len(returns1)), returns1)
                plt.plot(range(len(returns2)), returns2)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            np.save(self.save_path + '/returns1', returns1)
            np.save(self.save_path + '/returns2', returns2)
            np.save(self.save_path + '/seeds', seeds)
            np.save(self.save_path + '/args', self.args)
            if time_step > 0 and (time_step + 1) % self.args.evaluate_rate == 0 and self.args.save_loss:
                if not os.path.exists(self.save_path + '/loss'):
                    os.mkdir(self.save_path + '/loss')
                for agent_id, agent in enumerate(self.agents):
                    np.save(self.save_path + f'/loss/actor_loss{agent_id}', agent.policy.actor_loss)
                    np.save(self.save_path + f'/loss/critic_loss{agent_id}', agent.policy.critic_loss)

    def evaluate(self):
        returns1 = []
        returns2 = []
        pulls = {}
        rewards = {}
        coops = {}
        agent_names = ['adversary_0', 'agent_0']
        save_actions = {}
        save_positions = {}
        gazer = Gaze(self.args.gaze_type, self.env)
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s, _ = self.env.reset()
            s = [s[agent_names[0]], s[agent_names[1]]]
            gaze_actions = [0, 0]
            for i, state in enumerate(s):
                #print(f"EVALUATE: Agent {i} raw observation shape from env: {state.shape}")
                if self.args.lever_cue != 'none':
                    s[i] = np.concatenate((state, [0, 0])) #reward_cue, lever_cue
                    #print(f"EVALUATE: Agent {i} shape after adding reward_cue, lever_cue: {s[i].shape}")
                else:
                    s[i] = np.concatenate((state, [0])) #reward_cue
                    #print(f"EVALUATE: Agent {i} shape after adding reward_cue: {s[i].shape}")
                
                if self.args.lever_action: #NEW_CODE
                    s[i] = np.concatenate((s[i], [0]))  # lever_actions
                    s[i] = np.concatenate((s[i], [0])) #time_since_pull #NEW_CODE
                    #print(f"EVALUATE: Agent {i} shape after adding lever_action: {s[i].shape}")
                
            h = Updater.init_hidden(64)
            c = Updater.init_hidden(64)

            updater = Updater(self.args, self.env)
            #print(f"Evaluate: Agent {0} observation shape after updater: {s[0].shape}")
            rewards1 = 0
            rewards2 = 0
            eps_actions = {0: [], 1: []}
            eps_positions = {0: [], 1: []}
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = {}
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        #print(f"EVALUATE: Agent {agent_id} observation shape before gaze: {s[agent_id].shape}")
                        s[agent_id] = gazer.gaze(s[agent_id], gaze_actions[agent_id], agent_id)
                        if self.args.obfu is not None:
                            rand = np.random.uniform()
                            if rand < self.args.obfu:
                                s[agent_id][8] = 0 
                                s[agent_id][9] = 0 
                            #print(f"Agent {agent_id} observation shape after obfu: {s[agent_id].shape}")
                                # print("obfuscated")
                        #print(f"Agent {agent_id} observation shape before select_action: {s[agent_id].shape}")
                        action, h[agent_id], c[agent_id] = agent.select_action(s[agent_id], h[agent_id], c[agent_id], 0, 0)
                        actions[agent_names[agent_id]], gaze_actions[agent_id] = Updater.probs_to_actions(action, self.args.lever_action)
                        
                        eps_actions[agent_id].append((actions[agent_names[agent_id]], gaze_actions[agent_id]))
                        eps_positions[agent_id].append(s[agent_id][2])
                s_next, r, done, _, info = self.env.step(actions)
                if r == 100:
                    print("got_reward")
                s_next = [s_next[agent_names[0]], s_next[agent_names[1]]]
                r, s_next = updater.update(s_next, time_step, actions, gaze_actions)
                #print(f"Evaluate: s_next shapes after updater.update: {[s.shape for s in s_next]}")
                if r[0] == 100:
                    print("got_reward")
                rewards1 += r[0]
                rewards2 += r[1]
                s = s_next
                #print(f"Evaluate: s shapes after assignment: {[s[i].shape for i in range(len(s))]}")
            returns1.append(rewards1)
            returns2.append(rewards2)
            if True: # self.args.evaluate:
                pulls[episode] = updater.all_pulls
                rewards[episode] = updater.all_rewards
                coops[episode] = updater.all_coop
                save_actions[episode] = eps_actions
                save_positions[episode] = eps_positions
                # could also save first pulls, coop pulls, and lever cues if you wanted too..
        if self.args.evaluate:
            return pulls, rewards, save_actions, save_positions, coops
        else:
            return sum(returns1) / self.args.evaluate_episodes, sum(returns2) / self.args.evaluate_episodes, save_actions[episode][0].count(3), save_actions[episode][1].count(3)

