import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="test_forgot_to_name", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=20000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    # my additional arguments!!
    parser.add_argument("--reward-fn", type=str, default="coord", help="can be neg, buff, or coord")
    parser.add_argument("--buff", type=float, default=0.1, help="reward buffer for the big reward")
    parser.add_argument("--reward-value", type=int, default=100, help="reward value for a success")
    parser.add_argument("--load-weights", type=bool, default=False, help="load old set of weights or not")
    parser.add_argument("--load-name", type=str, default=None, help="weights to load")
    parser.add_argument("--method", type=str, default='maddpg', help="marl method (either maddpg or ica)")
    parser.add_argument("--threshold", type=int, default=4, help='cooperation threshold!')
    parser.add_argument("--refract", type=int, default=3, help="refractory period between pulls") # TODO: EVENTUALLY ASK WILL ABT THIS
    parser.add_argument("--low", type=int, default=2, help="smallest time to wait before lever comes out") # TODO: EVENTUALLY ASK WILL ABT THIS
    parser.add_argument("--high", type=int, default=8, help="largest time to wait before lever comes out") # TODO: EVENTUALLY ASK WILL ABT THIS
    parser.add_argument("--fail-low", type=int, default=2, help="smallest time to wait before lever comes out") # TODO: EVENTUALLY ASK WILL ABT THIS
    parser.add_argument("--fail-high", type=int, default=8, help="largest time to wait before lever comes out") # TODO: EVENTUALLY ASK WILL ABT THIS
    parser.add_argument("--num-actions", type=int, default=3, help="3 if 1D or 5 if 2D")
    parser.add_argument("--render-mode", type=str, default='None', help="None or human if you want to render env")
    parser.add_argument("--lever-cue", type=str, default='normal', help="normal, backin, or none")
    parser.add_argument("--actor-type", type=str, default='pytorch', help="can be linear, recurrent, or lstm")
    parser.add_argument("--critic-type", type=str, default='recurrent', help="can be linear, recurrent, or lstm")
    parser.add_argument("--gaze-type", type=str, default='full', help="can be full or partial")
    parser.add_argument("--obfu", type=float, default=None, help="rate of random obfuscation of the position of the other")
    parser.add_argument("--save-loss", type=bool, default=False, help="whether to save training loss or not...!")
    parser.add_argument("--run-num", type=int, default=99, help="what saved params of the model to load")
    parser.add_argument("--lever-action", type=bool, default=False, help="whether this is an additional lever action or not")
    parser.add_argument("--gaze-punishment", type=float, default=0, help="gaze punishment")
    parser.add_argument("--small-env", type=bool, default=True, help="small env or not")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=1, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=1000, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
