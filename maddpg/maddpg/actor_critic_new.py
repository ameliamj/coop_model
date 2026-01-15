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

class PEActor(nn.Module):
    def __init__(self, args, agent_id):
        super(PEActor, self).__init__()
        self.hidden_size = 64
        input_size = args.obs_shape[agent_id] + args.embed_shape[agent_id]
        output_size = args.action_shape[agent_id]

        self.lstm = nn.LSTMCell(input_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, embed, hidden_state, cell_state):
        inp = torch.cat((x, embed), dim=1)
        hidd, cell = self.lstm(inp, (hidden_state, cell_state))
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

class RECritic(nn.Module):
    def __init__(self, args):
        super(RECritic, self).__init__()
        self.hidden_size = 64
        input_size = sum(args.obs_shape) + sum(args.action_shape) + args.embed_shape[0]
        output_size = 1
        self.i2i = nn.Linear(input_size, self.hidden_size).to(device)
        self.i2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2o = nn.Linear(self.hidden_size, output_size)
    
    def forward(self, state, action, embed, hidden_state, c):  
        # embed = embed.reshape(x.shape[0], -1)
        
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action, embed], dim=1).to(device)
        
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        hidden_state = F.relu(self.h2h(hidden_state))
        hidden_state = torch.sigmoid(inp + hidden_state)
        output = self.h2o(hidden_state)
        return output, hidden_state, c

# this is a INDEPENDENT recurrent critic that has an input embedding!
class RECritic3(nn.Module):
    def __init__(self, args):
        super(RECritic3, self).__init__()
        self.hidden_size = 64
        input_size = args.obs_shape[0] + args.action_shape[0] + args.embed_shape[0]
        output_size = 1
        self.i2i = nn.Linear(input_size, self.hidden_size).to(device)
        self.i2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.h2o = nn.Linear(self.hidden_size, output_size)
    
    def forward(self, state, action, embed, hidden_state, c):  
        x = torch.cat([state, action, embed], dim=1).to(device)
        
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

class Embed(nn.Module):
    def __init__(self, args):
        super(Embed, self).__init__()
        agent_id = 0 # this doesn't really matter
        self.in_size = args.obs_shape[agent_id]
        self.hid_size = 64
        self.out_size = args.embed_shape[agent_id]
        self.embed_loss = args.embed_loss
        self.num_actions = args.num_actions

        self.layer_in = nn.Linear(self.in_size, self.hid_size).to(device)
        self.layer_mid = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.layer_dle = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.layer_out = nn.Linear(self.hid_size, self.out_size).to(device)

    def forward(self, obsv, hidd):
        x = F.relu(self.layer_in(obsv))
        x = F.relu(self.layer_mid(x))
        x = F.relu(self.layer_dle(x))
        # x = self.layer_out(x)
        x = torch.tanh(self.layer_out(x))
        # if self.out_size != 1:
        #     if self.embed_loss == 'action':
        #         x = torch.cat((F.softmax(x[:, :self.num_actions], dim=1), F.softmax(x[:, -2:], dim=1)), dim=1)
        #     elif self.embed_loss == 'move_action':
        #         x = F.softmax(x, dim=1)
        return x, hidd

class Embed_Baseline(nn.Module):
    def __init__(self, args):
        super(Embed_Baseline, self).__init__()
        agent_id = 0
        self.in_size = args.obs_shape[agent_id]
        self.hid_size = 64
        self.out_size = args.embed_shape[agent_id]

    def forward(self, obsv, hidd):
        embed = torch.zeros((obsv.shape[0], self.out_size))
        return embed, torch.zeros((hidd.shape)) # will just return an emtpy embedding to be used as a placeholder

class REmbed(nn.Module):
    def __init__(self, args):
        super(REmbed, self).__init__()
        agent_id = 0
        self.in_size = args.obs_shape[agent_id]
        self.hid_size = 64
        self.out_size = args.embed_shape[agent_id]
        self.embed_loss = args.embed_loss
        self.num_actions = args.num_actions
        
        self.i2i = nn.Linear(self.in_size, self.hid_size).to(device)
        self.i2h = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.h2h = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.h2o = nn.Linear(self.hid_size, self.out_size).to(device)

    def forward(self, obsv, hidd):
        x = obsv.to(device)
        hidden_state = hidd.to(device)
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        hidden_state = F.relu(self.h2h(hidden_state))
        hidden_state = torch.sigmoid(inp + hidden_state)
        actions = torch.tanh(self.h2o(hidden_state))
        # if self.out_size != 1:
        #     if self.embed_loss == 'action':
        #         x = torch.cat((F.softmax(actions[:, :self.num_actions], dim=1), F.softmax(actions[:, -2:], dim=1)), dim=1)
        #     elif self.embed_loss == 'move action':
        #         x = F.softmax(actions, dim=1)
        return actions, hidden_state

class SeqEmbed(nn.Module):
    def __init__(self, args):
        super(SeqEmbed, self).__init__()
        agent_id = 0
        self.in_size = args.obs_shape[agent_id]
        self.hid_size = 64
        self.out_size = args.action_shape[0] - 2 # this would be if we want to predict the policy for movement actions
        self.embed_loss = args.embed_loss
        self.num_actions = args.num_actions
        self.k = args.k
        
        self.i2i = nn.Linear(self.in_size, self.hid_size).to(device)
        self.i2h = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.h2h = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.h2o = nn.Linear(self.hid_size, self.out_size).to(device)

        self.norm = nn.LayerNorm(self.hid_size)

    def forward(self, obsv, hidd):
        x = obsv.to(device)
        hidden_state = hidd.to(device)
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        outputs = torch.empty(0)
        for i in range(self.k):
            hidden_state = self.norm(hidden_state)
            hidden_state = F.relu(self.h2h(hidden_state))
            if i == 0: # deciding to save the first hidden state! but could change this later...!
                first_hid = hidden_state
            hidden_state = torch.sigmoid(inp + hidden_state)
            actions = torch.tanh(self.h2o(hidden_state)) # this will be policy pred for that timestep
            outputs = torch.cat((outputs, actions), dim=1) # batch x (k x num actions)
        
        # outputs are now a list of policy predictions for k timesteps... DON'T KNOW THE DIMS THOUGH maybe (batch, k, 3)? where 3 is the policy dim?
        
        outputs = outputs.reshape(-1, self.out_size) # (batch x k) x num actions
        probs = F.softmax(torch.tensor(outputs, dtype=torch.float32), dim=-1) # (batch x k) x num actions
        a = torch.multinomial(probs, num_samples=1) # this should be a vector of k timesteps of just the action prediction or (batch, k)
        a = a.reshape(-1, self.k)
        outputs = outputs.reshape(-1, self.k, self.out_size)

        return (outputs, a), first_hid # then need to change this everywhere!! ("this" being how you are unpacking the tuple)
        # return actions, hidden_state

class SEmbed(nn.Module):
    def __init__(self, args):
        super(SEmbed, self).__init__()
        agent_id = 0
        self.in_size = args.obs_shape[agent_id]
        self.hid_size = 64
        self.out_size = 3 + 1 # this is action policy plus position, seems easier to hardcode rn
        self.embed_loss = args.embed_loss
        self.num_actions = args.num_actions
        self.args = args
        
        self.i2i = nn.Linear(self.in_size, self.hid_size).to(device)
        self.i2h = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.h2h = nn.Linear(self.hid_size, self.hid_size).to(device)
        self.h2o = nn.Linear(self.hid_size, self.out_size).to(device)

    def forward(self, obsv, hidd):
        
        x = obsv.to(device)
        hidden_state = hidd.to(device)
        x = F.relu(self.i2i(x))
        inp = F.relu(self.i2h(x))
        # this might be where you put the loop in...?
        outputs = torch.empty(0)
        for i in range(self.args.special_future): # or 5...?
            hidden_state = F.relu(self.h2h(hidden_state))
            if i == 0: # deciding to save the first hidden state! but could change this later...!
                first_hid = hidden_state
            # hidden_state = torch.sigmoid(inp + hidden_state)
            hidden_state = F.relu(inp + hidden_state)
            out = torch.tanh(self.h2o(hidden_state)).unsqueeze(1)
            # out = F.relu(self.h2o(hidden_state))
            outputs = torch.cat((outputs, out), dim=1)

        if not self.args.embed_run: # this means we are not training the embedding so we want it to be flat and action not policy!!
            outputs = outputs.reshape(-1, 4)
            probs = F.softmax(torch.tensor(outputs[:, :3], dtype=torch.float32), dim=-1)
            a = torch.multinomial(probs, num_samples=1)
            new_out = torch.cat([a, outputs[:, -1:]], dim=1)
            new_out = new_out.reshape(-1, 3, 2)
            # here, the shape is (batch, special_future, 2) 
            # if the shape embedding is only trained on one kind of loss, we want to zero out the other category...
            if self.args.special_act == 0: # only keep positions!
                new_out[:, :, 0] = torch.zeros((new_out.shape[0], self.args.special_future))
            if self.args.special_pos == 0: # only keep actions!
                new_out[:, :, 1] = torch.zeros((new_out.shape[0], self.args.special_future))
            outputs = new_out.reshape(new_out.shape[0], -1)
        return outputs, first_hid

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
