#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:08:11 2026

@author: david
"""

import torch
import os
import numpy as np
import torch.nn.functional as F
from maddpg.actor_critic import PActor
from maddpg.actor_critic3 import Critic3

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class IAC:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        self.actor_loss = []
        self.critic_loss = []

        # Create actor networks (Standard PyTorch Actor)
        self.actor_network = PActor(args, agent_id)
        self.actor_target_network = PActor(args, agent_id)
        
        # Create critic networks (Independent Critic)
        # Note: Critic3 is used here as an independent critic for this specific agent
        self.critic_network = Critic3(args, agent_id)
        self.critic_target_network = Critic3(args, agent_id)

        # Load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # Create the optimizers
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # Create directory for saving models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        
        self.model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
            
        self.model_path = os.path.join(self.model_path, 'agent_%d' % agent_id)
        if not os.path.exists(self.model_path) and not args.evaluate:
            os.mkdir(self.model_path)

        # Load existing weights if specified
        if args.load_weights and args.load_name is not None:
            load_path = self.args.save_dir[:7] + '/' + args.load_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_'
            self.actor_network.load_state_dict(torch.load(load_path + 'actor_params.pkl', weights_only=True))
            self.critic_network.load_state_dict(torch.load(load_path + 'critic_params.pkl', weights_only=True))
            print(f'Agent {self.agent_id} successfully loaded weights from {load_path}')

    def _soft_update_target_network(self):
        """Update target network parameters: θ_target = τ*θ + (1-τ)*θ_target"""
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def train(self, transitions, other_agents, timestep):
        """
        Train the Independent Actor-Critic model.
        Note: other_agents is included in signature for compatibility but ignored in IAC.
        """
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(device)

        # In IAC, we only care about this specific agent's data
        r = transitions['r_%d' % self.agent_id]
        o = transitions['o_%d' % self.agent_id]
        u = transitions['u_%d' % self.agent_id]
        o_next = transitions['o_next_%d' % self.agent_id]
        
        # Hidden and Cell states for recurrent components (Actor)
        ha = transitions['ha_%d' % self.agent_id]
        ca = transitions['ca_%d' % self.agent_id]
        ha_next = transitions['ha_next_%d' % self.agent_id]
        ca_next = transitions['ca_next_%d' % self.agent_id]

        # Hidden and Cell states for recurrent components (Critic)
        hc = transitions['hc_%d' % self.agent_id]
        cc = transitions['cc_%d' % self.agent_id]
        hc_next = transitions['hc_next_%d' % self.agent_id]
        cc_next = transitions['cc_next_%d' % self.agent_id]

        # --- 1. Calculate Target Q Value ---
        with torch.no_grad():
            # Get the next action from this agent's target actor
            u_next, _, _ = self.actor_target_network(o_next, ha_next, ca_next)
            
            # Get the next Q value from this agent's target critic (Independent: only uses self data)
            q_next, _, _ = self.critic_target_network(o_next, u_next, hc_next, cc_next)
            
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # --- 2. Update Critic ---
        # Current Q value estimation
        q_value, _, _ = self.critic_network(o, u, hc, cc)
        critic_loss = F.mse_loss(q_value, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # --- 3. Update Actor ---
        # Reselect action using current actor network
        u_current, _, _ = self.actor_network(o, ha, ca)
        
        # Policy gradient: maximize Q value (minimize -Q)
        actor_loss = -self.critic_network(o, u_current, hc, cc)[0].mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Log losses
        if self.args.save_loss and timestep % 100 == 0:
            self.actor_loss.append(actor_loss.detach().cpu().item())
            self.critic_loss.append(critic_loss.detach().cpu().item())

        # Update target networks
        self._soft_update_target_network()
        
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        torch.save(self.actor_network.state_dict(), os.path.join(model_path, num + '_actor_params.pkl'))
        torch.save(self.critic_network.state_dict(), os.path.join(model_path, num + '_critic_params.pkl'))