import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda:0")
device = torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, 64).to(device)
        self.action_out = nn.Linear(64, args.action_shape[agent_id]).to(device)

    def forward(self, x, fake_hidden):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions, fake_hidden
        
class RActor(nn.Module):
    def __init__(self, args, agent_id):
        super(RActor, self).__init__()
        self.hidden_size = 64
        input_size = args.obs_shape[agent_id]
        output_size = args.action_shape[agent_id]
        self.i2i = nn.Linear(input_size, self.hidden_size).to(device)
        self.i2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2o = nn.Linear(self.hidden_size, output_size).to(device)
    
    def forward(self, x, hidden_state, c):
        print('actor')
        print(hidden_state)
        print(hidden_state.shape)
        
        x = x.to(device)
        hidden_state = hidden_state.to(device)
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        hidden_state = F.relu(self.h2h(hidden_state))
        hidden_state = torch.sigmoid(inp + hidden_state)
        actions = torch.tanh(self.h2o(hidden_state))
        return actions, hidden_state, c

class LActor(nn.Module):
    def __init__(self, args, agent_id):
        super(LActor, self).__init__()
        self.hidden_size = 64
        self.input_size = args.obs_shape[agent_id]
        self.output_size = args.action_shape[agent_id]
        self.input = nn.Linear(self.input_size, self.hidden_size).to(device)

        # LSTM part...?
        # self.middle = nn.LSTM()
        self.inp1 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd1 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.inp2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.inp3 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd3 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.inp4 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd4 = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        self.output = nn.Linear(self.hidden_size, self.output_size).to(device)

    def forward(self, x, hidden_state, cell_state):
        # h_o, c_o = hidden_state
        h_o, c_o = hidden_state, cell_state
        x = F.relu(self.input(x))
        temp = self.inp1(x)
        temp2 = self.hidd1(h_o)
        # print(temp.shape)
        # print(temp2.shape)
        i = torch.sigmoid(self.inp1(x) + self.hidd1(h_o))
        f = torch.sigmoid(self.inp2(x) + self.hidd2(h_o))
        g = torch.tanh(self.inp3(x) + self.hidd3(h_o))
        o = torch.sigmoid(self.inp4(x) + self.hidd4(h_o))
        c = torch.mul(f, c_o) + torch.mul(i, g)
        h = torch.mul(o, torch.tanh(c))
        actions = torch.tanh(self.output(h))

        return actions, h, c

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, 64).to(device)
        self.q_out = nn.Linear(64, 1).to(device)

    def forward(self, state, action, fake_hidden):
        state = torch.cat(state, dim=1).to(device)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1).to(device)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value, fake_hidden

class RCritic(nn.Module):
    def __init__(self, args):
        super(RCritic, self).__init__()
        self.hidden_size = 64
        input_size = sum(args.obs_shape) + sum(args.action_shape)
        output_size = 1
        self.i2i = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), self.hidden_size).to(device)
        self.i2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2o = nn.Linear(self.hidden_size, output_size)
    
    def forward(self, state, action, hidden_state, c):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        hidden_state = torch.cat(hidden_state, dim=1).to(device)
        x = torch.cat([state, action], dim=1).to(device)
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        hidden_state = F.relu(self.h2h(hidden_state))
        hidden_state = torch.sigmoid(inp + hidden_state)
        output = self.h2o(hidden_state)
        hidden = hidden_state.reshape(2, 1, -1)
        hidden = hidden[0], hidden[1]
        return output, hidden, c

class LCritic(nn.Module):
    def __init__(self, args):
        super(LCritic, self).__init__()
        self.hidden_size = 64
        self.input_size = sum(args.obs_shape) + sum(args.action_shape)
        self.output_size = 1
        self.input = nn.Linear(self.input_size, self.hidden_size).to(device)

        # LSTM part...?
        self.inp1 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd1 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.inp2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.inp3 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd3 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.inp4 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.hidd4 = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        self.output = nn.Linear(self.hidden_size, self.output_size).to(device)

    def forward(self, state, action, hidden_state, cell_state):
        # h_o, c_o = hidden_state
        h_o, c_o = hidden_state, cell_state
        h_o = torch.cat(h_o, dim=1).to(device)
        c_o = torch.cat(c_o, dim=1).to(device)

        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1).to(device)

        x = F.relu(self.input(x))
        i = torch.sigmoid(self.inp1(x) + self.hidd1(h_o))
        f = torch.sigmoid(self.inp2(x) + self.hidd2(h_o))
        g = torch.tanh(self.inp3(x) + self.hidd3(h_o))
        o = torch.sigmoid(self.inp4(x) + self.hidd4(h_o))
        c = torch.mul(f, c_o) + torch.mul(i, g)
        h = torch.mul(o, torch.tanh(c))
        output = torch.tanh(self.output(h))

        h = h.reshape(2, 1, -1)
        c = c.reshape(2, 1, -1)
        h = h[0], h[1]
        c = c[0], c[1]

        return output, h, c


# JUST NOT GOING TO TOUCH THIS RIGHT NOW
"""class ICA_Critic(nn.Module):
    def __init__(self, args, agent_id):
        super(ICA_Critic, self).__init__()
        self.max_action = args.high_action
        # self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc1 = nn.Linear(args.obs_shape[agent_id] + args.action_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value"""
