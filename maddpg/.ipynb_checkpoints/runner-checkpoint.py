from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from updater import Updater
from gaze import Gaze
from maddpg.actor_critic import Embed # TODO: TAKE THIS OUT
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

            # Reset environment at start of each episode
            if time_step % self.episode_limit == 0:
                seed = np.random.randint(0, 1000)
                seeds.append(seed)
                s, _ = self.env.reset(seed=seed)
                s = [s[agent_names[0]], s[agent_names[1]]]
                for i, state in enumerate(s):
                    if self.args.lever_cue != 'none':
                        s[i] = np.concatenate((state, [0, 0]))
                    else:
                        s[i] = np.concatenate((state, [0]))
                ha = Updater.init_hidden(hidden_size=64)
                ha_next = Updater.init_hidden(hidden_size=64)
                hc = Updater.init_hidden(hidden_size=64)
                hc_next = Updater.init_hidden(hidden_size=64)
                ca = Updater.init_hidden(hidden_size=64)
                ca_next = Updater.init_hidden(hidden_size=64)
                cc = Updater.init_hidden(hidden_size=64)
                cc_next = Updater.init_hidden(hidden_size=64)
                he = Updater.init_hidden(hidden_size=64)
                he_next = Updater.init_hidden(hidden_size=64)
                
                updater = Updater(self.args, self.env)
            
            # Select actions for both agents
            u = []
            embeddings = []
            actions = {}
            gaze_actions = [0, 0]
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    s[agent_id] = gazer.gaze(s[agent_id], gaze_actions[agent_id], agent_id)

                    if self.args.embed_input != 'none':
                        embed, he_next[agent_id] = agent.get_embed(s[agent_id], he[agent_id])
                    if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
                        action, ha_next[agent_id], ca_next[agent_id] = agent.select_action(s[agent_id], ha[agent_id], ca[agent_id], embed, self.noise, self.epsilon)
                    else:
                        action, ha_next[agent_id], ca_next[agent_id] = agent.select_action(s[agent_id], ha[agent_id], ca[agent_id], embed=None, noise_rate=self.noise, epsilon=self.epsilon)

                    u.append(action)
                    if self.args.embed_input != 'none':
                        embeddings.append(embed)
                    actions[agent_names[agent_id]], gaze_actions[agent_id] = Updater.probs_to_actions(action, self.args.lever_action)

                for agent_id, agent in enumerate(self.agents):
                    if self.args.embed_input == 'critic' or self.args.embed_input == 'both':
                        temp_val, hc_next[agent_id], cc_next[agent_id] = agent.get_value(s[:self.args.n_agents], u, hc[agent_id], cc[agent_id], embed, agent_id=agent_id)
                    else:
                        temp_val, hc_next[agent_id], cc_next[agent_id] = agent.get_value(s[:self.args.n_agents], u, hc[agent_id], cc[agent_id], embed=None, agent_id=agent_id)

            # Do action
            s_next, r, done, _, info = self.env.step(actions)
            s_next = [s_next[agent_names[0]], s_next[agent_names[1]]]

            # Update rewards, coord_times, and cues
            r, s_next = updater.update(s_next, time_step, actions, gaze_actions)
            # Update buffer and save results
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], ha[:self.args.n_agents], ha_next[:self.args.n_agents], hc[:self.args.n_agents], hc_next[:self.args.n_agents], ca[:self.args.n_agents], ca_next[:self.args.n_agents], cc[:self.args.n_agents], cc_next[:self.args.n_agents], embeddings, he[:self.args.n_agents], he_next[:self.args.n_agents])
            s = s_next
            ha = ha_next
            hc = hc_next
            ca = ca_next
            cc = cc_next
            he = he_next
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
                    if self.args.embed_train != 'none' or self.args.embed_test != 'none':
                        np.save(self.save_path + f'/loss/embed_loss{agent_id}', agent.policy.embed_loss)

    def evaluate(self):
        returns1 = []
        returns2 = []
        pulls = {}
        rewards = {}
        coops = {}
        agent_names = ['adversary_0', 'agent_0']
        save_actions = {}
        save_positions = {}
        embeds = {}
        gazer = Gaze(self.args.gaze_type, self.env)
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s, _ = self.env.reset()
            s = [s[agent_names[0]], s[agent_names[1]]]
            gaze_actions = [0, 0]
            for i, state in enumerate(s):
                if self.args.lever_cue != 'none':
                    s[i] = np.concatenate((state, [0, 0]))
                else:
                    s[i] = np.concatenate((state, [0]))
            h = Updater.init_hidden(64)
            c = Updater.init_hidden(64)

            he = Updater.init_hidden(64)
            updater = Updater(self.args, self.env)
            rewards1 = 0
            rewards2 = 0
            eps_actions = {0: [], 1: []}
            eps_positions = {0: [], 1: []}
            eps_embeds = np.zeros((self.args.evaluate_episode_len, 4, 5))
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = {}
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        s[agent_id] = gazer.gaze(s[agent_id], gaze_actions[agent_id], agent_id)
                        if self.args.obfu is not None:
                            rand = np.random.uniform()
                            if rand < self.args.obfu:
                                s[agent_id][8] = 0 # self.bad
                                s[agent_id][9] = 0 # self.bad
                                # print("obfuscated")

                        if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
                            embed, he[agent_id] = agent.get_embed(s[agent_id], he[agent_id])
                            # if agent_id == 1:
                            #     print(embed, F.softmax(torch.tensor(action[:3])), F.softmax(torch.tensor(action[-2:])))
                            action, h[agent_id], c[agent_id] = agent.select_action(s[agent_id], h[agent_id], c[agent_id], embed, 0, 0)
                        else:
                            action, h[agent_id], c[agent_id] = agent.select_action(s[agent_id], h[agent_id], c[agent_id], None, 0, 0)
                        actions[agent_names[agent_id]], gaze_actions[agent_id] = Updater.probs_to_actions(action, self.args.lever_action)

                        if not (self.args.embed_test == 'None'): # need to do this in better more array like way!!
                            
                            embed = agent.policy.embed_actor(h[agent_id])
                            eps_embeds[time_step, agent_id, :] = embed
                            eps_embeds[time_step, agent_id + 2, :] = action
                        else:
                            if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
                                eps_embeds[time_step, agent_id, :] = embed
                                eps_embeds[time_step, agent_id + 2, :] = action
                            elif self.args.embed_input == 'critic':
                                embed, he[agent_id] = agent.get_embed(s[agent_id], he[agent_id])
                                eps_embeds[time_step, agent_id, :] = embed
                                eps_embeds[time_step, agent_id + 2, :] = action
                        
                        
                        eps_actions[agent_id].append((actions[agent_names[agent_id]], gaze_actions[agent_id]))
                        eps_positions[agent_id].append(s[agent_id][2])
                s_next, r, done, _, info = self.env.step(actions)
                if r == 100:
                    print("got_reward")
                s_next = [s_next[agent_names[0]], s_next[agent_names[1]]]
                r, s_next = updater.update(s_next, time_step, actions, gaze_actions)
                if r[0] == 100:
                    print("got_reward")
                rewards1 += r[0]
                rewards2 += r[1]
                s = s_next
            returns1.append(rewards1)
            returns2.append(rewards2)
            if True: # self.args.evaluate:
                pulls[episode] = updater.all_pulls
                rewards[episode] = updater.all_rewards
                coops[episode] = updater.all_coop
                save_actions[episode] = eps_actions
                save_positions[episode] = eps_positions
                if not (self.args.embed_test is None):
                    embeds[episode] = eps_embeds
                # could also save first pulls, coop pulls, and lever cues if you wanted too..
        if self.args.evaluate:
            return pulls, rewards, save_actions, save_positions, coops, embeds
        else:
            return sum(returns1) / self.args.evaluate_episodes, sum(returns2) / self.args.evaluate_episodes, save_actions[episode][0].count(3), save_actions[episode][1].count(3)

    # please ignore this, this is trying to train embeddings online, 
    # not that interested in testing this right now
    def embed_evaluate(self):
        returns1 = []
        returns2 = []
        pulls = {}
        rewards = {}
        coops = {}
        agent_names = ['adversary_0', 'agent_0']
        save_actions = {}
        save_positions = {}
        gazer = Gaze(self.args.gaze_type, self.env)

        embed_lr = 1e-5
        embedder1 = Embed(self.args)
        embedder2 = Embed(self.args)
        embed_optim1 = torch.optim.Adam(embedder1.parameters(), lr=embed_lr)
        embed_optim2 = torch.optim.Adam(embedder2.parameters(), lr=embed_lr)
        embed_losses1 = []
        embed_losses2 = []

        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s, _ = self.env.reset()
            s = [s[agent_names[0]], s[agent_names[1]]]
            # gaze_action = 0 # None
            gaze_actions = [0, 0]
            for i, state in enumerate(s):
                if self.args.lever_cue:
                    s[i] = np.concatenate((state, [0, 0]))
                else:
                    s[i] = np.concatenate((state, [0]))
            h = Updater.init_hidden(64)
            c = Updater.init_hidden(64)

            he = Updater.init_hidden(64)
            updater = Updater(self.args, self.env)
            rewards1 = 0
            rewards2 = 0
            # pulls = {0: [], 1: []}
            # rewards = {0: [], 1: []}
            eps_actions = {0: [], 1: []}
            eps_positions = {0: [], 1: []}
            # eps_positions = {0: np.zeros((self.args.evaluate_episode_len, 2)), 1: np.zeros((self.args.evaluate_episode_len, 2))}
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = {}
                embeds = []
                u = []
                if True:  # with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        s[agent_id] = gazer.gaze(s[agent_id], gaze_actions[agent_id], agent_id)
                        # rand = np.random.uniform()
                        # if np.random.uniform() < self.args.obfu:
                        #     s[agent_id][8] = 0 # self.bad
                        #     s[agent_id][9] = 0 # self.bad
                        #     # print("obfuscated")

                        inputs = torch.tensor(s[agent_id], dtype=torch.float32).unsqueeze(0)
                        if agent_id == 0:

                            embed, _ = embedder1(inputs, he[agent_id])
                            embeds.append(embed)
                        else:
                            embed, _ = embedder2(inputs, he[agent_id])
                            embeds.append(embed)
                        empty_embed = torch.zeros((1, 5))
                        with torch.no_grad():
                            action, h[agent_id], c[agent_id] = agent.select_action(s[agent_id], h[agent_id],
                                                                                   c[agent_id], empty_embed, 0, 0)

                            u.append(action)
                            actions[agent_names[agent_id]], gaze_actions[agent_id] = Updater.probs_to_actions(action, self.args.lever_action)

                        eps_actions[agent_id].append((actions[agent_names[agent_id]], gaze_actions[agent_id]))
                        eps_positions[agent_id].append(s[agent_id][2])

                # train embedder
                if self.args.embed_input != 'none':
                    e_loss_fn = torch.nn.CrossEntropyLoss()
                    embed_loss1 = e_loss_fn(embeds[0], F.softmax(torch.tensor(u[1][:self.args.num_actions])).reshape(1, -1))
                    embed_loss2 = e_loss_fn(embeds[1], F.softmax(torch.tensor(u[0][:self.args.num_actions])).reshape(1, -1))

                    embed_optim1.zero_grad()
                    embed_loss1.backward()
                    embed_optim1.step()
                    embed_optim2.zero_grad()
                    embed_loss2.backward()
                    embed_optim2.step()

                    embed_losses1.append(embed_loss1.detach())
                    embed_losses2.append(embed_loss2.detach())

                s_next, r, done, _, info = self.env.step(actions)
                s_next = [s_next[agent_names[0]], s_next[agent_names[1]]]
                r, s_next = updater.update(s_next, time_step, actions, gaze_actions)
                rewards1 += r[0]
                rewards2 += r[1]
                s = s_next
            returns1.append(rewards1)
            returns2.append(rewards2)
            if True:  # self.args.evaluate:
                pulls[episode] = updater.all_pulls
                rewards[episode] = updater.all_rewards
                coops[episode] = updater.all_coop
                save_actions[episode] = eps_actions
                save_positions[episode] = eps_positions
        plt.figure()
        plt.plot(range(len(embed_losses1)), embed_losses1)
        plt.plot(range(len(embed_losses2)), embed_losses2)
        plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/plt.png', format='png')
        torch.save(embedder1.state_dict(), self.args.save_dir + '/' + 1 + '_embed_params.pkl')
        torch.save(embedder2.state_dict(), self.args.save_dir + '/' + 2 + '_embed_params.pkl')
        if self.args.evaluate:
            return pulls, rewards, save_actions, save_positions, coops
        else:
            # print(f'number of gaze actions{pulls[episode][0].count(3), pulls[episode][1].count(3)}')
            return sum(returns1) / self.args.evaluate_episodes, sum(returns2) / self.args.evaluate_episodes, \
            save_actions[episode][0].count(3), save_actions[episode][1].count(3)

