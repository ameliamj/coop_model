import os
import numpy as np

root_dir = 'maddpg/model/'

def get_model_folds(model):
    folders = []
    for entry in os.scandir(root_dir + model):
        if entry.is_dir():
            folders.append(entry.name)
    if '.ipynb_checkpoints' in folders:
        folders.remove('.ipynb_checkpoints')
    folders = sorted(folders)
    return folders

def get_trained_models(folders, model):
    good_folds = {}
    ep_len = 75
    
    
    for fold in folders: 
        if 'evaluate' not in fold:
            try:
                args = np.load(root_dir + model + fold + '/args.npy', allow_pickle=True).item()
                good_reward = -1000 + ((args.reward_value + 1) * 18) # -800
                ret1 = np.load(f'{root_dir + model + fold}/returns1.npy')
                ret2 = np.load(f'{root_dir + model + fold}/returns2.npy')
            
                i = 0
                max_reward = -np.inf
                best_model = 0
                while i < len(ret1) - ep_len:
                    avg1 = np.mean(ret1[i: i+ep_len])
                    avg2 = np.mean(ret2[i: i+ep_len])
                    if avg1 > good_reward and avg2 > good_reward:
                        # print(i, avg1, avg2)
                        if np.mean((avg1, avg2)) > max_reward:
                            max_reward = np.mean((avg1, avg2))
                            best_model = i // 20
                    i += 20
                if best_model != 0:
                    # print(fold, round(max_reward, 2), best_model)
                    good_folds[fold] = (round(max_reward, 2), best_model)
                    if False:
                        plt.plot(ret1)
                        plt.plot(ret2)
                        plt.show()
            except:
                print(f'cant load {root_dir + model + fold}/returns1.npy')
    np.save(root_dir + '/' + model + '/good_folds', good_folds)
    return good_folds

def get_sim_commands(good_folds, model):
    out_dir = root_dir + model[:-1] + '/evaluate/'
    # make directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        # print("uncomment later")
        
    outpath = '.' + out_dir[6:]
    command_lines = ''
    for fold in good_folds.keys():
        args = np.load(root_dir + model + fold + '/args.npy', allow_pickle=True).item()
        gaze = args.gaze_type 
        embed_input = args.embed_input
        embed_model = args.embed_model
        reward = args.reward_value 
        scenario_name = fold + '_evaluate'
        load_name = model + fold
        run_num = 2 # good_folds[fold][1]
        eps = args.max_episode_len
        method = args.method
        embed_test = args.embed_test
        
        start_command = 'module load miniconda; conda activate social_comp; cd /home/aj764/project/grlt_next/grlt_next/maddpg;'
        end_command = f"python main.py --reward-fn=coord --method={method} --scenario-name={scenario_name} --threshold=7 --evaluate=True --max-episode-len={eps} --evaluate-episodes=1000 --evaluate-episode-len={eps} --gaze-type={gaze} --actor-type=pytorch --critic-type=recurrent --lever-cue=True --embed-input={embed_input} --embed-model={embed_model} --embed-test={embed_test} --save-dir={outpath} --reward-value={reward} --load-weights=True --load-name={load_name} --run-num={run_num}" 
        if 'small_env' in vars(args) and args.small_env:
            end_command += " --small-env=True"
        command_lines += (start_command + end_command + '\n')
        # print(start_command + end_command)
    return command_lines

def _get_sim_commands(good_folds, model):
    out_dir = root_dir + model[:-1] + '/evaluate/'
    # make directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        # print("uncomment later")
        
    outpath = '.' + out_dir[6:]
    command_lines = ''
    for fold in good_folds.keys():
        args = np.load(root_dir + model + fold + '/args.npy', allow_pickle=True).item()
        gaze = args.gaze_type 
        embed = args.embed_input # args.embed
        reward = args.reward_value 
        embed_type = args.embed_model # args.embed_type
        scenario_name = fold + '_evaluate'
        load_name = model + fold
        run_num = good_folds[fold][1]
        embed_test = args.embed_test
        
        
        start_command = 'module load miniconda; conda activate social_comp; cd /home/aj764/project/grlt_next/maddpg;'
        end_command = f"python main.py --reward-fn=coord --method=maddpg --scenario-name={scenario_name} --threshold=7 --evaluate=True --evaluate-episodes=1000 --evaluate-episode-len=1000 --gaze-type={gaze} --actor-type=pytorch --critic-type=recurrent --lever-cue=True --embed-input={embed} --embed-model={embed_type} --embed-test={embed_test} --save-dir={outpath} --reward-value={reward} --load-weights=True --load-name={load_name} --run-num={run_num}"
        command_lines += (start_command + end_command + '\n')
        # print(start_command + end_command)
    return command_lines