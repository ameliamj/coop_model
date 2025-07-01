import torch
import os
from maddpg.actor_critic import Actor, Critic, RActor, RCritic, LActor, LCritic

# device = torch.device("cuda:0")
device = torch.device("cpu")

# this is old, and I'm not 100% sure that it works. Also isn't the most updated with new actor/critic models
# and doesn't have anything on the embeddings
class ICA:
    def __init__(self, args, agent_id):  # Because different agents may have different obs and act dimensions, so the neural networks are different, agent_id is needed to distinguish

        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create actor networks
        if self.args.actor_type == 'reccurent':
            self.actor_network = RActor(args, agent_id) 
            self.actor_target_network = RActor(args, agent_id)
        elif self.args.actor_type == 'lstm':
            self.actor_network = LActor(args, agent_id) 
            self.actor_target_network = LActor(args, agent_id)
        else:
            self.actor_network = Actor(args, agent_id) 
            self.actor_target_network = Actor(args, agent_id)

        # create critic networks
        if self.args.critic_type == 'recurrent':
            self.critic_network = RCritic(args)
            self.critic_target_network = RCritic(args)
        elif self.args.critic_type == 'lstm':
            self.critic_network = LCritic(args)
            self.critic_target_network = LCritic(args)
        else:
            self.critic_network = Critic(args)
            self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path) and not args.evaluate:
            os.mkdir(self.model_path)

        if args.load_weights and args.load_name is not None:
            load_path = self.args.save_dir + '/' + args.load_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_'
            self.actor_network.load_state_dict(torch.load(load_path + 'actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(load_path + 'critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(device)
        r = transitions['r_%d' % self.agent_id]  # Only need your own reward during training
        o, u, o_next = [], [], []  # Used to store each agent's experience
        ha, ha_next = [], []
        hc, hc_next = [], []
        for agent_id in range(self.args.n_agents):
            # THIS CHANGE WILL MEAN THE SELF AGENT IS COPIED TWICE 
            # SO ALL ARRAY DIMENSIONS WILL STAY THE SAME BUT NO 
            # INFO FROM OTHER AGENT WILL BE TAKEN
            o.append(transitions['o_%d' % self.agent_id])
            u.append(transitions['u_%d' % self.agent_id])
            o_next.append(transitions['o_next_%d' % self.agent_id])
            
            ha.append(transitions['ha_%d' % self.agent_id])
            ha_next.append(transitions['ha_next_%d' % self.agent_id])
            hc.append(transitions['hc_%d' % self.agent_id])
            hc_next.append(transitions['hc_next_%d' % self.agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # Get the action corresponding to the next state
            index = 0
            for agent_id in range(self.args.n_agents):
                temp_action, temp_hidden = self.actor_target_network(o_next[agent_id], ha_next[agent_id])
                u_next.append(temp_action)
            q_next, hidden_temp = self.critic_target_network(o_next, u_next, hc_next)
            q_next = q_next.detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value, hidden_temp = self.critic_network(o, u, hc)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # Reselect the action of the current agent in the joint action, and the actions of other agents remain unchanged
        u[self.agent_id], temp_hidden = self.actor_network(o[self.agent_id], ha[self.agent_id])
        q_value, hidden_temp = self.critic_network(o, u, hc)
        actor_loss = - q_value.mean()
        if self.agent_id == 0:
            print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


