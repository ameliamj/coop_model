import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def forward(self, x, fake_hidden, c):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions, fake_hidden, c
        
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

        # input gate
        self.ii = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.hi = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # forget gate
        self.iff = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.hf = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # cell gate
        self.ig = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.hg = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # output gate
        self.io = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.ho = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # real output layer?
        self.oo = nn.Linear(self.hidden_size, self.output_size).to(device)
        
    def forward(self, x, hidden_state, cell_state):
        h_o, c_o = hidden_state, cell_state

        i_t = torch.sigmoid(self.ii(x) + self.hi(h_o))
        f_t = torch.sigmoid(self.iff(x) + self.hf(h_o))
        g_t = torch.tanh(self.ig(x) + self.hg(h_o))
        o_t = torch.sigmoid(self.io(x) + self.ho(h_o))

        c_t = f_t * c_o + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        out = torch.tanh(self.oo(h_t))

        return out, h_t, c_t

class PActor(nn.Module):
    def __init__(self, args, agent_id):
        super(PActor, self).__init__()
        self.hidden_size = 64
        input_size = args.obs_shape[agent_id]
        output_size = args.action_shape[agent_id]

        self.lstm = nn.LSTMCell(input_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        hidd, cell = self.lstm(x, (hidden_state, cell_state))
        out = torch.tanh(self.output(hidd))
        return out, hidd, cell

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, 64).to(device)
        self.q_out = nn.Linear(64, 1).to(device)

    def forward(self, state, action, fake_hidden, c):
        state = torch.cat(state, dim=1).to(device)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1).to(device)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value, fake_hidden, c

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
        x = torch.cat([state, action], dim=1).to(device)
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        hidden_state = F.relu(self.h2h(hidden_state))
        hidden_state = torch.sigmoid(inp + hidden_state)
        output = self.h2o(hidden_state)
        return output, hidden_state, c

class LCritic(nn.Module):
    def __init__(self, args):
        super(LCritic, self).__init__()
        self.hidden_size = 64
        self.input_size = sum(args.obs_shape) + sum(args.action_shape)
        self.output_size = 1
        
        # input gate
        self.ii = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.hi = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # forget gate
        self.iff = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.hf = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # cell gate
        self.ig = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.hg = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # output gate
        self.io = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.ho = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # real output layer?
        self.oo = nn.Linear(self.hidden_size, self.output_size).to(device)

    def forward(self, state, action, hidden_state, cell_state):
        h_o, c_o = hidden_state, cell_state

        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1).to(device)

        i_t = torch.sigmoid(self.ii(x) + self.hi(h_o))
        f_t = torch.sigmoid(self.iff(x) + self.hf(h_o))
        g_t = torch.tanh(self.ig(x) + self.hg(h_o))
        o_t = torch.sigmoid(self.io(x) + self.ho(h_o))

        c_t = f_t * c_o + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        out = torch.tanh(self.oo(h_t))

        return out, h_t, c_t

class PCritic(nn.Module):
    def __init__(self, args):
        super(PCritic, self).__init__()
        self.hidden_size = 64
        self.input_size = sum(args.obs_shape) + sum(args.action_shape)
        self.output_size = 1

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, state, action, hidden_state, cell_state):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1).to(device)
        
        hidd, cell = self.lstm(x, (hidden_state, cell_state))
        out = torch.tanh(self.output(hidd))
        return out, hidd, cell
