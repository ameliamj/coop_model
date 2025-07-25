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


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self):
        print("Make World Small")
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
        print("Reset World Small")
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
                landmark.state.p_pos += [0, 0.25]
            else:
                landmark.state.p_pos += [0, -0.25]
            if i < 2:
                landmark.state.p_pos += [-0.25, 0]
            else:
                landmark.state.p_pos += [0.25, 0]
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, agent in enumerate(world.agents):
            x_pos = float(np_random.uniform(-0.25, +0.25, 1))
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
        
        '''comm = [] #OLD_CODE
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
                2]] + other_pos + other_goal)'''
            
        #NEW_CODE
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos[0] - agent.state.p_pos[0])
            # other_goal.append(other.state.p_pos - other.goal_a.state.p_pos)
        if not agent.adversary:
            obsv = np.concatenate([agent.state.p_pos[0]] + [agent.state.p_vel[0]] + [agent.goal_a.state.p_pos[0] - agent.state.p_pos[0]] + [entity_pos[
                3][0]] + other_pos)
        else:
            obsv = np.concatenate([agent.state.p_pos[0]] + [agent.state.p_vel[0]] + [agent.goal_a.state.p_pos[0] - agent.state.p_pos[0]] + [entity_pos[
                2][0]] + other_pos)
        
        print("og obsv: ", obsv)
        
        return obsv

    # using the absolute positions of entities instead of relative position doesn't work! (using relative, agent-centric positions above)