from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
import time
import pickle


if __name__ == '__main__':
    print("STARTING")
    # get the params
    start = time.time()
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        all_pulls, all_rewards, all_actions, all_positions, all_coops = runner.evaluate()
        with open(args.save_dir + '/' + args.scenario_name + "/all_pulls.pkl", "wb") as f:
            pickle.dump(all_pulls, f)
        with open(args.save_dir + '/' + args.scenario_name + "/all_rewards.pkl", "wb") as f:
            pickle.dump(all_rewards, f)
        with open(args.save_dir + '/' + args.scenario_name + "/all_actions.pkl", "wb") as f:
            pickle.dump(all_actions, f)
        with open(args.save_dir + '/' + args.scenario_name + "/all_positions.pkl", "wb") as f:
            pickle.dump(all_positions, f)
        with open(args.save_dir + '/' + args.scenario_name + "/all_coops.pkl", "wb") as f:
            pickle.dump(all_coops, f)
    else:
        runner.run()
    print(time.time() - start)
