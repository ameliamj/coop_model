import numpy as np
import torch
from maddpg.maddpg import MADDPG

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        if True: # self.args.method == 'maddpg':
            self.policy = MADDPG(args, agent_id)

    def select_action(self, o, h, c, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
            hidden = h
            cell = c
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi, hidden, cell = self.policy.actor_network(inputs, h, c)
            pi = pi.squeeze(0)
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy(), hidden, cell

    def get_value(self, o, u, h, c, agent_id):
        o_input = []
        u_input = []
        for i, obsv in enumerate(o):
            o_input.append(torch.tensor(o[i], dtype=torch.float32).unsqueeze(0))# .reshape(1, -1))
            u_input.append(torch.tensor(u[i], dtype=torch.float32).unsqueeze(0))
        q_val, hidden, cell = self.policy.critic_network(o_input, u_input, h, c)
        q_val = q_val.squeeze(0)
        return_val = q_val.cpu().numpy()        
        return return_val.copy(), hidden, cell

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)



