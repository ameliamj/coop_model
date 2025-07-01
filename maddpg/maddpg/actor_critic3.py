import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# device = torch.device("cuda:0")
device = torch.device("cpu")

class Actor3(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor3, self).__init__()
        self.hidden_size = 64
        input_size = args.obs_shape[agent_id]
        output_size = args.action_shape[agent_id]

        self.lstm = nn.LSTMCell(input_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        hidd, cell = self.lstm(x, (hidden_state, cell_state))
        out = torch.tanh(self.output(hidd))
        return out, hidd, cell

class Critic3(nn.Module):
    def __init__(self, args, agent_id):
        super(Critic3, self).__init__()
        self.agent_id = agent_id
        self.hidden_size = 64
        input_size = args.obs_shape[0] + args.action_shape[0]
        out_size = 1
        self.i2i = nn.Linear(input_size, self.hidden_size).to(device)
        self.i2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2o = nn.Linear(self.hidden_size, out_size).to(device)
    
    def forward(self, state, action, hidden_state, c):
        # state = torch.cat(state, dim=1)
        # action = torch.cat(action, dim=1)
        # this is SUS as hell, I get that
        x = torch.cat([state, action], dim=1).to(device)
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        hidden_state = F.relu(self.h2h(hidden_state))
        hidden_state = torch.sigmoid(inp + hidden_state)
        output = self.h2o(hidden_state)
        return output, hidden_state, c

class Embed(nn.Module):
    def __init__(self, args):
        super(Embed, self).__init__()
        self.hid_size = 64
        self.out_size = args.embed_shape[0]
        self.embed_loss = args.embed_loss
        self.num_actions = args.num_actions

        self.layer_out = nn.Linear(self.hid_size, self.out_size).to(device)

    def forward(self, x):
        x = self.layer_out(x)
        if self.out_size != 1:
            if self.embed_loss == 'action':
                x = torch.cat((F.softmax(x[:, :self.num_actions], dim=1), F.softmax(x[:, -2:], dim=1)), dim=1)
            elif self.embed_loss == 'move_action':
                x = F.softmax(x, dim=1)
        return x