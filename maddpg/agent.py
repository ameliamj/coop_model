import numpy as np
import torch
from maddpg.maddpg import MADDPG
from maddpg.ica import ICA
from maddpg.maddpg3 import MADDPG3

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        if self.args.method == 'maddpg':
            self.policy = MADDPG(args, agent_id)
        elif self.args.method == 'maddpg3':
            self.policy = MADDPG3(args, agent_id)
        else: 
            self.policy = ICA(args, agent_id)

    def select_action(self, o, h, c, embed, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
            hidden = h
            cell = c
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            # inputs = o
            if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
                pi, hidden, cell = self.policy.actor_network(inputs, embed, h, c)
            else:
                pi, hidden, cell = self.policy.actor_network(inputs, h, c)
            pi = pi.squeeze(0)
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy(), hidden, cell

    def get_value(self, o, u, h, c, embed, agent_id):
        o_input = []
        u_input = []
        for i, obsv in enumerate(o):
            o_input.append(torch.tensor(o[i], dtype=torch.float32).unsqueeze(0))# .reshape(1, -1))
            u_input.append(torch.tensor(u[i], dtype=torch.float32).unsqueeze(0))
        if self.args.embed_input == 'critic' or self.args.embed_input == 'both':
            q_val, hidden, cell = self.policy.critic_network(o_input, u_input, embed, h, c)
        else:
            if self.args.embed_test is not None:
                q_val, hidden, cell = self.policy.critic_network(o_input[agent_id], u_input[agent_id], h, c)
            else:
                q_val, hidden, cell = self.policy.critic_network(o_input, u_input, h, c)
        q_val = q_val.squeeze(0)
        return_val = q_val.cpu().numpy()        
        return return_val.copy(), hidden, cell

    def get_embed(self, o, he):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        embed, he_next = self.policy.embed(inputs, he)
        return embed, he_next

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)



