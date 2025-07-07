import numpy as np
# observation structure is:
# agent position (0, 1); agent velocity (2, 3); relative agent lever (4, 5); relative agent reward port (6, 7); 
# relative other agent (8, 9); other agent relative other lever (10, 11); reward  cue (12); lever cue (13)

class Gaze:
    def __init__(self, gaze_type, env):
        self.gaze_type = gaze_type
        self.env = env
        self.bad = 0

    def gaze(self, state, gaze_action, agent_id):
        if self.gaze_type == 'full':
            return state
        elif self.gaze_type == 'partial':
            return self.partial_gaze(state, gaze_action, agent_id)
        elif self.gaze_type == 'none':
            return self.no_gaze(state, gaze_action, agent_id)
        else:
            raise NotImplementedError


    def no_gaze(self, state, gaze_action, agent_id):
        state[8] = self.bad
        state[9] = self.bad
        return state

    def partial_gaze(self, state, gaze_action, agent_id):
        # return state
        if gaze_action == 1:
            # choosing to observe other
            return state
        else:
            # not choosing to observe other 
            # print("returning bad state")
            state[8] = self.bad
            state[9] = self.bad
            return state