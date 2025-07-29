import torch
from torch.nn import functional as F
import numpy as np
import time

class Updater:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.pull_times = [-1, -1]
        self.lever_cues = [0, 0]
        self.reward_cues = [0, 0]
        self.lever_actions = [0, 0]
        self.first_pull = -1 - self.args.threshold
        if self.args.reward_fn == 'buff':
            self.waits = np.random.randint(self.args.low, self.args.high, 2)
        elif self.args.reward_fn == 'instrumental':
            self.waits = np.random.randint(self.args.low, self.args.high, 2)
        elif self.args.reward_fn == 'coord':
            rand_start = np.random.randint(self.args.low, self.args.high, 1)
            self.waits = [rand_start, rand_start]
            print("self.waits: ", self.waits)
        self.all_pulls = {0: [], 1: []} 
        self.all_rewards = {0: [], 1: []}
        self.all_coop = []
        self.first_pulls = []
        self.coop_pulls = []
        self.all_lever_cues = []

    def update_colors(self):
        for i, landmark in enumerate(self.env.unwrapped.world.landmarks):
            if i < 2:
                # lever
                if self.args.lever_cue and self.lever_cues[i] == 1:
                    # lever cue is on
                    landmark.color = np.array([0.7, 0.7, 0.1])
                else:
                    # lever cue is off
                    landmark.color = np.array([0.1, 0.1, 0.1])
                    landmark.color[i + 1] += 0.8
            else:
                # reward
                if self.reward_cues[i - 2] == 1:
                    # reward cue is on
                    landmark.color = np.array([0.7, 0.7, 0.1])
                else:
                    # reward cue is off
                    landmark.color = np.array([0.1, 0.1, 0.1])
                    landmark.color[i - 1] += 0.9

    def check_lever_pull(self, i, l, timestep, real_actions): #If Lever Cue, then it checks whether there has been a lever action and the levers are active and agent is close enough to lever and enough time has passed since last pull and agent is not exactly at lever's postiion
        lever_action = (self.args.lever_action == False) or (self.args.lever_action == True and real_actions[i] == 3)
        if self.args.lever_cue == 'none':
            return lever_action and l[i] > -self.args.buff and l[i] != 0 and (
                        timestep - self.pull_times[i]) > self.args.refract
        else:
            return lever_action and l[i] > -self.args.buff and l[i] != 0 and (
                    timestep - self.pull_times[i]) > self.args.refract and self.lever_cues[i] == 1


    def update(self, s_next, timestep, actions, gaze_actions):
        # lever cue can be normal, none, or back in
        # reward function can be buff or coord
        print("UPDATE")
        real_actions = []
        for agent in actions:
            real_actions.append(actions[agent])

        if self.args.reward_fn == 'buff':
            r, s_next = self.buff_update(s_next, timestep, real_actions, gaze_actions)
        elif self.args.reward_fn == 'pavlovian':
            r, s_next = self.pavlovian_update(s_next, timestep, real_actions, gaze_actions)
        elif self.args.reward_fn == 'instrumental':
            r, s_next = self.instrumental_update(s_next, timestep, real_actions, gaze_actions)
        else: #  self.args.reward_fn == 'coord'
            print("COORD UPDATE")
            r, s_next = self.coord_update(s_next, timestep, real_actions, gaze_actions)
        return r, s_next
    
    def pavlovian_update(self, s_next, timestep, real_actions, gaze_actions):
        a = 1

    def instrumental_update(self, s_next, timestep, real_actions, gaze_actions):
        r = -1 * np.ones(len(s_next)) #default reward (-1)              # OR np.zeros(len(s)) OR some distant measure from current target??
        l = np.zeros(len(s_next)) #lever distances
        p = np.zeros(len(s_next)) #reward distances

        for i, state in enumerate(s_next):
            # find the distance to the lever
            l[i] = -s_next[i][2]
            # find the distance to the reward port
            p[i] = -s_next[i][3]

            # check for lever pull
            if self.check_lever_pull(i, l, timestep, real_actions):
                self.lever_cues[i] = 0
                self.reward_cues[i] = 1
                self.pull_times[i] = timestep
                self.all_pulls[i].append(timestep)

            # check for reward
            if p[i] > -self.args.buff and self.reward_cues[i] == 1:
                r[i] = self.args.reward_value
                self.reward_cues[i] = 0
                self.waits[i] = timestep + np.random.randint(self.args.low, self.args.high, 1)
                self.all_rewards[i].append(timestep)

            # check for lever reset 
            if timestep == self.waits[i]:
                self.lever_cues[i] = 1

            # update lever/reward cues in state
            s_next[i] = np.concatenate((state, [self.reward_cues[i]]))
            if self.args.lever_cue != 'none':
                s_next[i] = np.concatenate((s_next[i], [self.lever_cues[i]]))
            if self.args.lever_action:
                if (timestep - self.pull_times[i]) < self.args.threshold:
                    self.lever_actions[i] = 1
                else:
                    self.lever_actions[i] = 0
                s_next[i] = np.concatenate((s_next[i], [self.lever_actions[i]]))
                
                # Compute normalized time since last pull (0 to 1)
                if self.pull_times[i] >= 0:
                    time_since_pull = (timestep - self.pull_times[i]) / self.args.threshold
                else:
                    time_since_pull = 0  # No pull yet
                s_next[i] = np.concatenate((s_next[i], [time_since_pull]))

            # negative reward for gaze
            if gaze_actions[i] == 1:
                r[i] += self.args.gaze_punishment
        self.update_colors()
        return r, s_next
    

    def buff_update(self, s_next, timestep, real_actions, gaze_actions):
        r = -1 * np.ones(len(s_next)) #default reward (-1)              # OR np.zeros(len(s)) OR some distant measure from current target??
        l = np.zeros(len(s_next)) #lever distances
        p = np.zeros(len(s_next)) #reward distances

        for i, state in enumerate(s_next):
            # find the distance to the lever
            l[i] = -np.sqrt(np.sum(np.square(s_next[i][4:6])))
            # find the distance to the reward port
            p[i] = -np.sqrt(np.sum(np.square(s_next[i][6:8])))

            # check for lever pull
            if self.check_lever_pull(i, l, timestep, real_actions):
                self.lever_cues[i] = 0
                self.reward_cues[i] = 1
                self.pull_times[i] = timestep
                self.all_pulls[i].append(timestep)

            # check for reward
            if p[i] > -self.args.buff and self.reward_cues[i] == 1:
                r[i] = self.args.reward_value
                self.reward_cues[i] = 0
                self.waits[i] = timestep + np.random.randint(self.args.low, self.args.high, 1)
                self.all_rewards[i].append(timestep)

            # check for lever reset (irrelevant if no lever cue)
            if timestep == self.waits[i]:
                self.lever_cues[i] = 1

            # update lever/reward cues in state
            s_next[i] = np.concatenate((state, [self.reward_cues[i]]))
            if self.args.lever_cue != 'none':
                s_next[i] = np.concatenate((s_next[i], [self.lever_cues[i]]))
            if self.args.lever_action:
                if (timestep - self.pull_times[i]) < self.args.threshold:
                    self.lever_actions[i] = 1
                else:
                    self.lever_actions[i] = 0
                s_next[i] = np.concatenate((s_next[i], [self.lever_actions[i]]))

            # negative reward for gaze
            if gaze_actions[i] == 1:
                r[i] += self.args.gaze_punishment
        self.update_colors()
        return r, s_next

    def coord_update(self, s_next, timestep, real_actions, gaze_actions):
        
        #print("\nRunning Coord Update")
        
        r = -1 * np.ones(len(s_next))  # OR np.zeros(len(s)) OR some distant measure from current target??
        l = np.zeros(len(s_next))
        p = np.zeros(len(s_next))

        #print("Init s_next: ", s_next)     
        #print("len(s_next): ", len(s_next))

        for i, state in enumerate(s_next):
            '''# find the distance to the lever #OLD_CODE
            l[i] = -np.sqrt(np.sum(np.square(s_next[i][4:6])))
            # find the distance to the reward port
            p[i] = -np.sqrt(np.sum(np.square(s_next[i][6:8])))'''
            
            
            #print("x_pos: ", s_next[i][0])
            
            # find the distance to the lever #NEW_CODE
            l[i] = -s_next[i][2]
            print("l[i]: ", l[i])
            # find the distance to the reward port
            p[i] = -s_next[i][3]
            print("p[i]: ", p[i])
            print("-self.args.buff: ", -self.args.buff)

            # check for pull
            pull_prod = np.prod(self.pull_times)
            if self.check_lever_pull(i, l, timestep, real_actions):
                self.pull_times[i] = timestep
                self.all_pulls[i].append(timestep)
                if pull_prod > 0 and np.prod(self.pull_times) < 0:
                    self.first_pull = timestep
                    self.first_pulls.append(timestep)

            # check for cooperation
            if np.sum(self.pull_times) > 0 and np.abs(self.pull_times[0] - self.pull_times[1]) < self.args.threshold:
                print("Cooperation Satisfied")
                self.lever_cues = [0, 0]
                self.reward_cues = [1, 1]
                self.all_coop.append((self.pull_times[0], self.pull_times[1]))
                self.pull_times = [-1, -1]
                self.coop_pulls.append(timestep)

            # check for reward
            if p[i] > -self.args.buff and self.reward_cues[i] == 1:
                print("Reward Gotten")
                r[i] = self.args.reward_value
                self.reward_cues[i] = 0
                self.all_rewards[i].append(timestep)

            # check for lever reset
            if self.args.lever_cue == 'backin':
                # lever back in if cooperation threshold passed
                if timestep == (self.first_pull + self.args.threshold):
                    self.lever_cues = [0, 0]
                    new_wait = timestep + np.random.randint(self.args.fail_low, self.args.fail_high, 1)
                    self.waits = [new_wait, new_wait]
                    self.pull_times = [-1, -1]
            # lever out after rewards
            if np.sum(self.lever_cues) == 0 and np.sum(self.reward_cues) == 0 and np.mean(self.waits) < timestep:
                new_wait = timestep + np.random.randint(self.args.low, self.args.high, 1)
                self.waits = [new_wait, new_wait]
            
            print("waitTime: ", np.mean(self.waits))
            if int(np.mean(self.waits)) == timestep and np.sum(self.reward_cues) == 0:
                self.lever_cues = [1, 1]
                self.all_lever_cues.append(timestep)

            # update lever/reward cues in state
            s_next[i] = np.concatenate((state, [self.reward_cues[i]])) #OLD_CODE
            if self.args.lever_cue != 'none':
                s_next[i] = np.concatenate((s_next[i], [self.lever_cues[i]]))
            if self.args.lever_action:
                if (timestep - self.pull_times[i]) < self.args.threshold:
                    self.lever_actions[i] = 1
                else:
                    self.lever_actions[i] = 0
                s_next[i] = np.concatenate((s_next[i], [self.lever_actions[i]]))
                #print(f"Agent {i} s_next shape: {s_next[i].shape}")  # Debug print
                
                #NEW_CODE: 
                # Compute normalized time since last pull (0 to 1)
                if self.pull_times[i] >= 0:
                    time_since_pull = (timestep - self.pull_times[i]) / self.args.threshold
                else:
                    time_since_pull = 0  # No pull yet
                s_next[i] = np.concatenate((s_next[i], [time_since_pull]))
                #NEW_CODE
                

            # negative reward for gaze
            if gaze_actions[i] == 1:
                r[i] += self.args.gaze_punishment
        self.update_colors()
        
        #print("End s_next: ", s_next)        
        #print("len(s_next): ", len(s_next))
        
        return r, s_next


    @staticmethod
    def probs_to_actions(action, lever_action=False):
        if lever_action:
            probs = F.softmax(torch.tensor(action[:4], dtype=torch.float32), dim=-1)
        else:
            probs = F.softmax(torch.tensor(action[:3], dtype=torch.float32), dim=-1)
        a = torch.multinomial(probs, num_samples=1)
        a = a.tolist()[0]

        gaze_probs = F.softmax(torch.tensor(action[-2:], dtype=torch.float32), dim=-1)
        g = torch.multinomial(gaze_probs, num_samples=1)
        g = g.tolist()[0]
        return a, g


    @staticmethod
    def init_hidden(hidden_size, lstm=False):
        if lstm:
            h = [torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size)), torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size))]
            c = [torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size)), torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size))]
            return [h, c]
        else:
            # return [torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size)), torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size))]
            return [torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size)), torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_size))]




