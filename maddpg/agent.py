import numpy as np
import torch
from maddpg.maddpg import MADDPG
from maddpg.iac_new import IAC

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        if self.args.method == 'maddpg':
            print("Maddpg Model")
            self.policy = MADDPG(args, agent_id)
        elif self.args.method == 'ica':
            print("IAC Model")
            self.policy = IAC(args, agent_id)
        else:
            raise ValueError(f"Unknown method: {self.args.method}")

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
        if self.args.method == 'maddpg':
            o_input = []
            u_input = []
            for i, obsv in enumerate(o):
                o_input.append(torch.tensor(o[i], dtype=torch.float32).unsqueeze(0))# .reshape(1, -1))
                u_input.append(torch.tensor(u[i], dtype=torch.float32).unsqueeze(0))
            q_val, hidden, cell = self.policy.critic_network(o_input, u_input, h, c)
            q_val = q_val.squeeze(0)
            return_val = q_val.cpu().numpy() 
            return return_val.copy(), hidden, cell
        else:
            # Independent input for IAC
            o_self = torch.tensor(o[self.agent_id], dtype=torch.float32).unsqueeze(0)
            u_self = torch.tensor(u[self.agent_id], dtype=torch.float32).unsqueeze(0)
            q_val, hidden, cell = self.policy.critic_network(o_self, u_self, h, c)
            
        return q_val.squeeze(0).cpu().numpy().copy(), hidden, cell
        

    def learn(self, transitions, other_agents, timestep = 0):
        # Pass the timestep for IAC compatibility
        if self.args.method == 'ica':
            self.policy.train(transitions, other_agents, timestep)
        else:
            self.policy.train(transitions, other_agents)



