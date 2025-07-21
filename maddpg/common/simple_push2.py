# noqa: D212, D415
"""
# Simple Push

```{figure} mpe_simple_push.gif
:width: 140px
:name: simple_push
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_push_v3` |
|--------------------|---------------------------------------------|
| Actions            | Discrete/Continuous                         |
| Parallel API       | Yes                                         |
| Manual Control     | No                                          |
| Agents             | `agents= [adversary_0, agent_0]`            |
| Agents             | 2                                           |
| Action Shape       | (5)                                         |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))             |
| Observation Shape  | (8),(19)                                    |
| Observation Values | (-inf,inf)                                  |
| State Shape        | (27,)                                       |
| State Values       | (-inf,inf)                                  |


This environment has 1 good agent, 1 adversary, and 1 landmark. The good agent is rewarded based on the distance to the landmark. The adversary is rewarded if it is close to the landmark, and if the agent is far from the landmark (the difference of the distances). Thus the adversary must learn to
push the good agent away from the landmark.

Agent observation space: `[self_vel, goal_rel_position, goal_landmark_id, all_landmark_rel_positions, landmark_ids, other_agent_rel_positions]`

Adversary observation space: `[self_vel, all_landmark_rel_positions, other_agent_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_push_v3.env(max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size


"""

import numpy as np
import gymnasium as gym
import pygame

from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        lever_action = False
    ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            # dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_push_v3"
        
        self.action_space = [gym.spaces.Discrete(4 if lever_action else 3) for _ in range(2)] #NEW_CODE
        
        #Rendering Setup: 
        '''self.window_size = 700  # Pygame window size
        self.world_scale = 350  # Scale world coords [-1, 1] to pixels
        self.rat_sprite = None
        self.rat_frames = []
        self.frame_count = 4  # Number of animation frames
        self.frame_index = 0
        self.frame_timer = 0
        self.frame_duration = 100  # Milliseconds per frame
        self.font = None '''
    '''  
    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            # Load rat sprite sheet (assumes 4 frames, 32x32 each, horizontal)
            try:
                self.rat_sprite = pygame.image.load("rat_sprite.png").convert_alpha()
                frame_width = self.rat_sprite.get_width() // self.frame_count
                for i in range(self.frame_count):
                    frame = self.rat_sprite.subsurface((i * frame_width, 0, frame_width, frame_width))
                    frame = pygame.transform.scale(frame, (30, 30))  # Scale to fit
                    self.rat_frames.append(frame)
            except FileNotFoundError:
                print("Error: rat_sprite.png not found in project directory")
                self.rat_frames = [pygame.Surface((30, 30)) for _ in range(self.frame_count)]  # Fallback
            # Initialize font
            self.font = pygame.font.SysFont("arial", 20)  # Use default font
            # self.font = pygame.font.Font("arial.ttf", 20)  # Uncomment if using custom font
        # Update animation frame
        self.frame_timer += 16  # Approx 60 FPS (16ms per frame)
        if self.frame_timer >= self.frame_duration:
            self.frame_index = (self.frame_index + 1) % self.frame_count
            self.frame_timer = 0
        # Clear screen
        self.screen.fill((255, 255, 255))  # White background
        # Draw landmarks
        for i, landmark in enumerate(self.world.landmarks):
            # Convert world coords [-1, 1] to screen coords [0, 700]
            x = int((landmark.state.p_pos[0] + 1) * self.world_scale)
            y = int((1 - landmark.state.p_pos[1]) * self.world_scale)  # Flip y-axis
            color = tuple(int(c * 255) for c in landmark.color)  # RGB 0-1 to 0-255
            # Draw landmark as circle
            pygame.draw.circle(self.screen, color, (x, y), 15)
            # Add text: "L" for levers (i=0,1), "R" for reward zones (i=2,3)
            text = "L" if i < 2 else "R"
            text_surface = self.font.render(text, True, (0, 0, 0))  # Black text
            text_rect = text_surface.get_rect(center=(x, y))
            self.screen.blit(text_surface, text_rect)
        # Draw agents
        for agent in self.world.agents:
            x = int((agent.state.p_pos[0] + 1) * self.world_scale)
            y = int((1 - agent.state.p_pos[1]) * self.world_scale)
            # Draw animated rat sprite
            self.screen.blit(self.rat_frames[self.frame_index], (x - 15, y - 15))  # Center sprite
        pygame.display.flip()
        # Handle Pygame events to allow window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                raise SystemExit'''
    

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 4
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = False # True CHANGING THIS BC
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.index = i
            if i < 2:
                landmark.color[i + 1] += 0.8
            else:
                landmark.color[i - 1] += 0.9
        # set goal landmark
        for i, agent in enumerate(world.agents):
            agent.goal_a = world.landmarks[i]
            agent.color = np.array([0.25, 0.25, 0.25])
            agent.color[i + 1] += 0.25
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.zeros(world.dim_p)
            if i % 2 == 0:
                landmark.state.p_pos += [0, 1]
            else:
                landmark.state.p_pos += [0, -1]
            if i < 2:
                landmark.state.p_pos += [-1, 0]
            else:
                landmark.state.p_pos += [1, 0]
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, agent in enumerate(world.agents):
            x_pos = float(np_random.uniform(-1, +1, 1))
            y_pos = agent.goal_a.state.p_pos[1]
            agent.state.p_pos = np.array([x_pos, y_pos])
            
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def agent_reward(self, agent, world):
        return 0 # reward calculated in "updater"

    def adversary_reward(self, agent, world):
        return 0 # reward calculated in "updater"

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        print("Observation")
        entity_pos = []
        print("world.landmarks: ", world.landmarks)
        for i, entity in enumerate(world.landmarks):  # world.entities:
            print(f"i: {i},  entity.state.p_pos: {entity.state.p_pos}")
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            print("entity.color: ", entity.color)
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        other_goal = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # other_goal.append(other.state.p_pos - other.goal_a.state.p_pos)
            other_goal.append(other.goal_a.state.p_pos - agent.state.p_pos) # everything is agent centric
        if not agent.adversary:
            obsv = np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + [agent.goal_a.state.p_pos - agent.state.p_pos] + [entity_pos[
                3]] + other_pos + other_goal)
        else:
            obsv = np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + [agent.goal_a.state.p_pos - agent.state.p_pos] + [entity_pos[
                2]] + other_pos + other_goal)
        return obsv

    # using the absolute positions of entities instead of relative position doesn't work! (using relative, agent-centric positions above)