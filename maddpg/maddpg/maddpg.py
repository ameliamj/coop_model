import torch
import os
import torch.nn.functional as F
from maddpg.actor_critic import Actor, Critic, RActor, RCritic, LActor, LCritic, PActor, PCritic
from maddpg.actor_critic import PEActor, RECritic, Embed, REmbed, Embed_Baseline

# device = torch.device("cuda:0")
device = torch.device("cpu")

class MADDPG:
    def __init__(self, args, agent_id):  # Because different agents may have different obs and act dimensions, so the neural networks are different, agent_id is needed to distinguish

        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        self.actor_loss = []
        self.critic_loss = []
        self.embed_loss = []

        # create actor networks
        if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
            self.actor_network = PEActor(args, agent_id) 
            self.actor_target_network = PEActor(args, agent_id)
            print("making an embedding actor")
        elif self.args.actor_type == 'recurrent':
            self.actor_network = RActor(args, agent_id) 
            self.actor_target_network = RActor(args, agent_id)
            print("making a recurrent actor")
        elif self.args.actor_type == 'lstm':
            self.actor_network = LActor(args, agent_id) 
            self.actor_target_network = LActor(args, agent_id)
            print("making an lstm actor")
        elif self.args.actor_type == 'pytorch':
            self.actor_network = PActor(args, agent_id) 
            self.actor_target_network = PActor(args, agent_id)
            print("making an pytorch actor")
        else:
            self.actor_network = Actor(args, agent_id) 
            self.actor_target_network = Actor(args, agent_id)

        # create critic networks
        if self.args.embed_input == 'critic' or self.args.embed_input == 'both':
            self.critic_network = RECritic(args)
            self.critic_target_network = RECritic(args)
            print("making an embedding critic")
        elif self.args.critic_type == 'recurrent':
            self.critic_network = RCritic(args)
            self.critic_target_network = RCritic(args)
            print("making a recurrent critic")
        elif self.args.critic_type == 'lstm':
            self.critic_network = LCritic(args)
            self.critic_target_network = LCritic(args)
            print("making an lstm critic")
        elif self.args.critic_type == 'pytorch':
            self.critic_network = PCritic(args)
            self.critic_target_network = PCritic(args)
            print("making an pytorch critic")
        else:
            self.critic_network = Critic(args)
            self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # maybe create an embedder
        if self.args.embed_input != 'none':
            if self.args.embed_model== 'baseline':
                self.embed = Embed_Baseline(args)
                self.embed_target_network = Embed_Baseline(args)
            elif self.args.embed_model== 'recurrent':
                self.embed = REmbed(args)
                self.embed_target_network = REmbed(args)
            else:
                self.embed = Embed(args)
                self.embed_target_network = Embed(args)
            if self.args.embed_train == 'offline':
                self.embed_target_network.load_state_dict(self.embed.state_dict())
                self.embed_optim = torch.optim.Adam(self.embed.parameters(), lr=self.args.lr_critic)

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

        if args.load_weights and args.load_name is not None: # TODO change back to 99 pls
            load_path = self.args.save_dir[:7] + '/' + args.load_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_' # '/99_'
            self.actor_network.load_state_dict(torch.load(load_path + 'actor_params.pkl', weights_only=True))
            self.critic_network.load_state_dict(torch.load(load_path + 'critic_params.pkl', weights_only=True))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          load_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           load_path + '/critic_params.pkl'))
        if args.embed_input != 'none' and args.load_embed and (args.load_name is not None or args.embed_name is not None):
            if args.embed_name is not None:
                load_path = self.args.save_dir[:7] + '/' + args.embed_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_'  # '/99_'
                self.embed.load_state_dict(torch.load(load_path + 'embed_params.pkl'))
                print('Agent {} successfully loaded embed: {}'.format(self.agent_id,
                                                                              load_path + '/embed_params.pkl'))
            elif args.load_name is not None:
                load_path = self.args.save_dir[:7] + '/' + args.load_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_'  # '/99_'
                self.embed.load_state_dict(torch.load(load_path + 'embed_params.pkl'))
                print('Agent {} successfully loaded embed: {}'.format(self.agent_id,
                                                                      load_path + '/embed_params.pkl'))
            else:
                raise Exception("you need to specify a path to load an embedding from...") # this should be impossible but its a check??

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        if self.args.embed_train == 'offline':
            for target_param, param in zip(self.embed_target_network.parameters(), self.embed.parameters()):
                target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(device)
        r = transitions['r_%d' % self.agent_id]  # Only need your own reward during training
        o, u, o_next = [], [], []  # Used to store each agent's experience
        ha, ha_next = [], []
        hc, hc_next = [], []
        ca, ca_next = [], []
        cc, cc_next = [], []
        he, he_next = [], []        
        e = []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            
            ha.append(transitions['ha_%d' % agent_id])
            ha_next.append(transitions['ha_next_%d' % agent_id])
            hc.append(transitions['hc_%d' % agent_id])
            hc_next.append(transitions['hc_next_%d' % agent_id])

            ca.append(transitions['ca_%d' % agent_id])
            ca_next.append(transitions['ca_next_%d' % agent_id])
            cc.append(transitions['cc_%d' % agent_id])
            cc_next.append(transitions['cc_next_%d' % agent_id])

            he.append(transitions['he_%d' % agent_id])
            he_next.append(transitions['he_next_%d' % agent_id])

            e.append(transitions['e_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # Get the action corresponding to the next state
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    if self.args.embed_input != 'none':
                        self_embed_next, temp_hidden = self.embed_target_network(o_next[agent_id], he_next[agent_id])
                    if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
                        temp_action, temp_hidden, temp_cell = self.actor_target_network(o_next[agent_id], self_embed_next, ha_next[agent_id], ca_next[agent_id])
                    else:
                        temp_action, temp_hidden, temp_cell = self.actor_target_network(o_next[agent_id], ha_next[agent_id], ca_next[agent_id])
                    u_next.append(temp_action)
                else:
                    if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
                        other_embed, temp_hidden = other_agents[index].policy.embed_target_network(o_next[agent_id], he_next[agent_id])
                        temp_action, temp_hidden, temp_cell = other_agents[index].policy.actor_target_network(o_next[agent_id], other_embed, ha_next[agent_id], ca_next[agent_id])    
                    else:
                        temp_action, temp_hidden, temp_cell = other_agents[index].policy.actor_target_network(o_next[agent_id], ha_next[agent_id], ca_next[agent_id])
                    u_next.append(temp_action)
                    index += 1
            if self.args.embed_input == 'critic' or self.args.embed_input == 'both':
                q_next, hidden_temp, cell_temp = self.critic_target_network(o_next, u_next, self_embed_next, hc_next[self.agent_id], cc_next[self.agent_id])
            else:
                q_next, hidden_temp, cell_temp = self.critic_target_network(o_next, u_next, hc_next[self.agent_id], cc_next[self.agent_id])
            q_next = q_next.detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        if self.args.embed_input == 'critic' or self.args.embed_input == 'both':
            q_value, hidden_temp, cell_temp = self.critic_network(o, u, e[self.agent_id], hc[self.agent_id], cc[self.agent_id])
        else:
            q_value, hidden_temp, cell_temp = self.critic_network(o, u, hc[self.agent_id], cc[self.agent_id])
        critic_loss = (target_q - q_value).pow(2).mean()

        if self.args.embed_input != 'none':
            self_embed, temp_hidden = self.embed(o[self.agent_id], he[self.agent_id])

        # the actor loss
        # Reselect the action of the current agent in the joint action, and the actions of other agents remain unchanged
        if self.args.embed_input == 'actor' or self.args.embed_input == 'both':
            u[self.agent_id], temp_hidden, temp_cell = self.actor_network(o[self.agent_id], self_embed, ha[self.agent_id], ca[self.agent_id])
        else:
            u[self.agent_id], temp_hidden, temp_cell = self.actor_network(o[self.agent_id], ha[self.agent_id], ca[self.agent_id])
        if self.args.embed_input == 'critic' or self.args.embed_input == 'both':
            q_value, hidden_temp, cell_temp = self.critic_network(o, u, self_embed, hc[self.agent_id], cc[self.agent_id])
        else:
            q_value, hidden_temp, cell_temp = self.critic_network(o, u, hc[self.agent_id], cc[self.agent_id])
        actor_loss = - q_value.mean()

        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # embed loss
        if self.args.embed_train == 'offline':
            e_loss_fn = torch.nn.CrossEntropyLoss()
            if self.args.embed_loss == 'action':
                u[self.agent_id - 1] = u[self.agent_id - 1].detach()
                other_action_probs = torch.cat(
                    (F.softmax(u[self.agent_id - 1][:, :self.args.num_actions], dim=1), F.softmax(u[self.agent_id - 1][:, -2:], dim=1)),
                    dim=1)
                # print(self_embed.shape)
                # print(other_action_probs.shape)
                print(self.args.num_actions)
                embed_loss = e_loss_fn(self_embed, other_action_probs)
            elif self.args.embed_loss == 'move action':
                u[self.agent_id - 1] = u[self.agent_id - 1].detach()
                other_action_probs = F.softmax(u[self.agent_id - 1][:, :self.args.num_actionså], dim=1)
                embed_loss = e_loss_fn(self_embed, other_action_probs)
            else:  # embed_loss == pos
                embed_loss = e_loss_fn(self_embed, o[self.agent_id][:, 8:9])
            self.embed_optim.zero_grad()
            embed_loss.backward()
            self.embed_optim.step()

        if self.args.save_loss:
            self.actor_loss.append(actor_loss.detach())
            self.critic_loss.append(critic_loss.detach())
            if self.args.embed_train == 'offline':
                self.embed_loss.append(embed_loss.detach())

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
        if self.args.embed_input != 'none' and self.args.embed_train != 'none':
            torch.save(self.embed.state_dict(), model_path + '/' + num + '_embed_params.pkl')



