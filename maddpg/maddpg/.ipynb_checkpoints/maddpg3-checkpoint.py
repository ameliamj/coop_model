import torch
import os
import torch.nn.functional as F
from maddpg.actor_critic3 import Actor3, Critic3,  Embed

# device = torch.device("cuda:0")
device = torch.device("cpu")

# there are basically 3 different versions:
    # we have a loss function that is shared between the actor and embedding
    # we have a loss function that is shared between the critic and embedding
    # we have a loss function that is shared between the all three??
    # or could we have:
        # a weighted loss function between critic / embedding
        # a weighted loss function between actor / embedding
        # so that these two are potentially learning different embeddings??
        # they could both share the same layer for embedding to categorical distribution? but that actually makes no sense

class MADDPG3:
    def __init__(self, args, agent_id):  # Because different agents may have different obs and act dimensions, so the neural networks are different, agent_id is needed to distinguish

        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        self.actor_loss = []
        self.critic_loss = []
        self.embed_loss = [[], []]

        # create actor networks
        self.actor_network = Actor3(args, agent_id)
        self.actor_target_network = Actor3(args, agent_id)

        # create critic networks
        self.critic_network = Critic3(args, agent_id)
        self.critic_target_network = Critic3(args, agent_id)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        # self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # maybe create an embedder
        if args.embed_test == 'actor' or args.embed_test == 'both': # you really only need to create these IF you will train them...
            # in this case they are only for training...
            self.embed_actor = Embed(args)
        if args.embed_test == 'critic' or args.embed_test == 'both':
            self.embed_critic = Embed(args)

        if (args.embed_test == 'actor' or args.embed_test == 'both'): #  and args.embed_train == 'offline':
            self.embed_actor_optim = torch.optim.Adam(self.embed_actor.parameters(), lr=self.args.lr_critic)
        if (args.embed_test == 'critic' or args.embed_test == 'both'): #  and args.embed_train == 'offline':
            self.embed_critic_optim = torch.optim.Adam(self.embed_critic.parameters(), lr=self.args.lr_critic)

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

        # if args.load_weights:
        #     raise Exception("can't load weights with integrated embeddings right now")

        # THERE ARE FAR TO MANY NETWORKS TO FUFF AROUND WITH LOADING WEIGHTS RIGHT NOW!!
        if args.load_weights and args.load_name is not None: # TODO change back to 99 pls
            load_path = self.args.save_dir[:7] + '/' + args.load_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_' # '/99_'
            self.actor_network.load_state_dict(torch.load(load_path + 'actor_params.pkl', weights_only=True))
            self.critic_network.load_state_dict(torch.load(load_path + 'critic_params.pkl', weights_only=True))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          load_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           load_path + '/critic_params.pkl'))
        if args.load_embed and args.embed_name is not None:
            if args.embed_test == 'actor' or args.embed_test == 'both': # you really only need to create these IF you will train them...
                self.embed_actor.load_state_dict(torch.load(load_path + 'embed_actor_params.pkl', weights_only=True))
            if args.embed_test == 'critic' or args.embed_test == 'both':
                self.embed_critic.load_state_dict(torch.load(load_path + 'embed_critic_params.pkl', weights_only=True))
        # if args.embed_test != 'none' and args.load_embed and (args.load_name is not None or args.embed_name is not None):
        #     if args.embed_name is not None:
        #         load_path = self.args.save_dir[:7] + '/' + args.embed_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_'  # '/99_'
        #         self.embed.load_state_dict(torch.load(load_path + 'embed_params.pkl'))
        #         print('Agent {} successfully loaded embed: {}'.format(self.agent_id,
        #                                                                       load_path + '/embed_params.pkl'))
        #     elif args.load_name is not None:
        #         load_path = self.args.save_dir[:7] + '/' + args.load_name + '/' + 'agent_%d' % agent_id + '/' + str(args.run_num) + '_'  # '/99_'
        #         self.embed.load_state_dict(torch.load(load_path + 'embed_params.pkl'))
        #         print('Agent {} successfully loaded embed: {}'.format(self.agent_id,
        #                                                               load_path + '/embed_params.pkl'))
        #     else:
        #         raise Exception("you need to specify a path to load an embedding from...") # this should be impossible but its a check??

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
        ca, ca_next = [], []
        cc, cc_next = [], []
        he, he_next = [], []        
        e = []

        torch.autograd.set_detect_anomaly(True)
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
                    temp_action, temp_hidden, temp_cell = self.actor_target_network(o_next[agent_id], ha_next[agent_id], ca_next[agent_id])
                    u_next.append(temp_action)
                else:
                    temp_action, temp_hidden, temp_cell = other_agents[index].policy.actor_target_network(o_next[agent_id], ha_next[agent_id], ca_next[agent_id])
                    u_next.append(temp_action)
                    index += 1
            q_next, critic_target_hidden, critic_target_cell = self.critic_target_network(o_next[self.agent_id], u_next[self.agent_id], hc_next[self.agent_id], cc_next[self.agent_id])
            q_next = q_next.detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        first_q_value, critic_hidden, critic_cell = self.critic_network(o[self.agent_id], u[self.agent_id], hc[self.agent_id], cc[self.agent_id])
        critic_loss = (target_q - first_q_value).pow(2).mean()

        # the actor loss
        # Reselect the action of the current agent in the joint action, and the actions of other agents remain unchanged
        # we are fine not updating this in the typical way too because we are not using the full action space of both agents
        # u[self.agent_id], actor_hidden, actor_cell = self.actor_network(o[self.agent_id], ha[self.agent_id], ca[self.agent_id])
        new_action, actor_hidden, actor_cell = self.actor_network(o[self.agent_id], ha[self.agent_id], ca[self.agent_id])
        q_value, critic_hidden2, critic_cell2 = self.critic_network(o[self.agent_id], new_action, hc[self.agent_id], cc[self.agent_id])
        actor_loss = - q_value.mean()

        # find embed components of loss
        e_loss_fn = torch.nn.CrossEntropyLoss()
        # critic embed loss
        embed_critic_loss = 0
        if (self.args.embed_test == 'critic' or self.args.embed_test == 'both'): #  and self.args.embed_train == 'offline':
            critic_embed = self.embed_critic(critic_hidden)
            embed_critic_loss = self.get_embed_loss(critic_embed, u, o, e_loss_fn) # maybe this u being changed by actor is messed up!!!
        new_critic_loss = self.args.critic_alpha * critic_loss + self.args.critic_beta * embed_critic_loss

        # actor embed loss
        embed_actor_loss = 0
        if (self.args.embed_test == 'actor' or self.args.embed_test == 'both'): #  and self.args.embed_train == 'offline':
            actor_embed = self.embed_actor(actor_hidden)
            embed_actor_loss = self.get_embed_loss(actor_embed, u, o, e_loss_fn)
        actor_loss = self.args.actor_alpha * actor_loss + self.args.actor_beta * embed_actor_loss

        # update the network
        self.actor_optim.zero_grad()
        if (self.args.embed_test == 'actor' or self.args.embed_test == 'both'):
            self.embed_actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        if (self.args.embed_test == 'actor' or self.args.embed_test == 'both'):
            self.embed_actor_optim.step()

        self.critic_optim.zero_grad()
        if (self.args.embed_test == 'critic' or self.args.embed_test == 'both'):
            self.embed_critic_optim.zero_grad()
        new_critic_loss.backward(retain_graph=True)
        self.critic_optim.step()
        if (self.args.embed_test == 'critic' or self.args.embed_test == 'both'):
            self.embed_critic_optim.step()

        if self.args.save_loss:
            self.actor_loss.append(actor_loss.detach())
            self.critic_loss.append(critic_loss.detach())
            if self.args.embed_test == 'critic' or self.args.embed_test == 'both':
                self.embed_loss[1].append(embed_critic_loss.detach())
            if self.args.embed_test == 'actor' or self.args.embed_test == 'both':
                self.embed_loss[0].append(embed_actor_loss.detach())

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
        if self.args.embed_test == 'critic' or self.args.embed_test == 'both':
            torch.save(self.embed_critic.state_dict(), model_path + '/' + num + '_embed_critic_params.pkl')
        if self.args.embed_test == 'actor' or self.args.embed_test == 'both':
            torch.save(self.embed_actor.state_dict(), model_path + '/' + num + '_embed_actor_params.pkl')

    def get_embed_loss(self, curr_embed, u, o, loss_fn):
        if self.args.embed_loss == 'action':
            u[self.agent_id - 1] = u[self.agent_id - 1].detach()
            other_action_probs = torch.cat(
                (F.softmax(u[self.agent_id - 1][:, :self.args.num_actions], dim=1),
                 F.softmax(u[self.agent_id - 1][:, -2:], dim=1)),
                dim=1)
            embed_loss = loss_fn(curr_embed, other_action_probs)
            print(curr_embed[0], other_action_probs[0])
        elif self.args.embed_loss == 'move action':
            u[self.agent_id - 1] = u[self.agent_id - 1].detach()
            other_action_probs = F.softmax(u[self.agent_id - 1][:, :self.args.num_actionså], dim=1)
            embed_loss = loss_fn(curr_embed, other_action_probs)
        else:  # embed_loss == pos
            embed_loss = loss_fn(curr_embed, o[self.agent_id][:, 8:9])
        return embed_loss


