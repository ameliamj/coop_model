import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.ndimage import uniform_filter1d
from scipy.stats import ttest_ind

from metrics import get_gaze, get_repress, get_reward, get_interpress, get_ratid2, get_actions

root_dir = './maddpg/model/'

def make_array(total_loss):
    min_t = np.inf
    for t in total_loss:
        if len(t) < min_t:
            min_t = len(t)
    
    for i, t in enumerate(total_loss):
        total_loss[i] = t[:min_t]
    
    return np.array(total_loss)

def graph_reward_functions(dirs):
    # get reward functions!
    for model_dir in dirs:
        good = np.load(root_dir + model_dir + '/good_folds.npy', allow_pickle=True).item()
        num_models = len(list(good.keys()))
        tot_models = len(os.listdir(root_dir + model_dir)) - 1
        eps = 2000 if model_dir != 'no_embed_full' else 1400

        all_ret1 = [] # np.zeros((num_models, eps))
        all_ret2 = [] # np.zeros((num_models, eps))
        for i, g in enumerate(good):
            ret1 = np.load(root_dir + model_dir + '/' + g + '/returns1.npy', allow_pickle=True)
            ret2 = np.load(root_dir + model_dir + '/' + g + '/returns2.npy', allow_pickle=True)

            all_ret1.append(ret1)
            all_ret2.append(ret2)
            # try:
            #     all_ret1[i] = ret1[:eps]
            #     all_ret2[i] = ret2[:eps]
            # except Exception as e:
            #     print(g, e)

        all_ret1 = make_array(all_ret1)
        all_ret2 = make_array(all_ret2)
        
        mean_ret1 = np.mean(all_ret1, axis=0)
        std_ret1 = np.std(all_ret1, axis=0)

        mean_ret2 = np.mean(all_ret2, axis=0)
        std_ret2 = np.std(all_ret2, axis=0)

        plt.plot(mean_ret1, color='tab:blue')
        plt.plot(mean_ret2, color='tab:orange')

        try:
            x = np.arange(all_ret1.shape[1])
            plt.fill_between(x, mean_ret1 - std_ret1, mean_ret1 + std_ret1, color='tab:blue', alpha=0.3)
            plt.fill_between(x, mean_ret2 - std_ret2, mean_ret2 + std_ret2, color='tab:blue', alpha=0.3)
    
            window = 50  # You can adjust this value for more or less smoothing
            smooth_ret1 = uniform_filter1d(mean_ret1, size=window, mode='nearest')
            smooth_ret2 = uniform_filter1d(mean_ret2, size=window, mode='nearest')
    
            plt.plot(smooth_ret1, color='k')
            plt.plot(smooth_ret2, color='k')
            plt.title(f'{model_dir}: {num_models} / {tot_models} models trained')
            plt.xlabel('training episodes')
            plt.ylabel('episode reward')
            plt.show()
        except:
            continue

def graph_gaze(dirs):
    # Example setup
    model_names = dirs
    num_dir = len(model_names)

    # Example data shape: each list contains arrays of shape (n_runs,)
    # Replace this with your actual data structure
    # Example: all_ret1_models[0] is a list of returns for Model A's condition 1

    all_ret1_models = []
    all_ret2_models = []

    for model_dir in dirs:
        good = np.load(root_dir + model_dir + '/good_folds.npy', allow_pickle=True).item()
        num_models = len(list(good.keys()))
        tot_models = 10 * (len(os.listdir(root_dir + model_dir)) // 10)
        eps = 2000 if model_dir != 'no_embed_full' else 1400

        all_ret1 = np.zeros(num_models)
        all_ret2 = np.zeros(num_models)
        for i, g in enumerate(good):
            filepath = root_dir + model_dir + '/evaluate/' + g + '_evaluate'
            gaze1, gaze2 = get_gaze(filepath)
            all_ret1[i] = gaze1
            all_ret2[i] = gaze2
        all_ret1_models.append(all_ret1)
        all_ret2_models.append(all_ret2)

    # Combine ret1 and ret2 for each model
    combined_returns = [np.concatenate([ret1, ret2]) for ret1, ret2 in zip(all_ret1_models, all_ret2_models)]

    # Compute means and standard errors
    means = [np.mean(ret) for ret in combined_returns]
    stds = [np.std(ret) for ret in combined_returns]
    ses = [np.std(ret) / np.sqrt(len(ret)) for ret in combined_returns]

    # Bar plot
    x = np.arange(num_dir)
    fig, ax = plt.subplots(figsize=(5, 6))

    bars = ax.bar(x, means, yerr=ses, capsize=5, color='skyblue', edgecolor='black')

    # Add individual dots (ret1 in blue, ret2 in orange)
    for i in range(num_dir):
        jitter = 0.08
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret1_models[i])), all_ret1_models[i], '.', color='tab:blue',
                alpha=0.6, label='Return 1' if i == 0 else "")
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret2_models[i])), all_ret2_models[i], '.',
                color='tab:orange', alpha=0.6, label='Return 2' if i == 0 else "")

    t_stat, p_val = ttest_ind(combined_returns[0], combined_returns[1])
    print(t_stat, p_val)

    # Final touches
    ax.set_xticks(x)
    ax.set_xticklabels(['with full gaze', 'with partial gaze'], rotation=45)
    ax.set_ylabel("gaze events")
    plt.tight_layout()
    plt.show()

def graph_repress(dirs):
    # %%
    # Example setup
    model_names = dirs
    num_dir = len(model_names)

    all_ret1_models = []
    all_ret2_models = []

    for model_dir in dirs:
        good = np.load(root_dir + model_dir + '/good_folds.npy', allow_pickle=True).item()
        if 'baseline_pause_13' in good:
            value = good.pop('baseline_pause_13', None)
        num_models = len(list(good.keys()))
        tot_models = 10 * (len(os.listdir(root_dir + model_dir)) // 10)
        eps = 2000 if model_dir != 'no_embed_full' else 1400

        all_ret1 = np.zeros(num_models)
        all_ret2 = np.zeros(num_models)
        for i, g in enumerate(good):
            filepath = root_dir + model_dir + '/evaluate/' + g + '_evaluate'
            reward1, reward2 = get_repress(filepath)

            all_ret1[i] = reward1
            all_ret2[i] = reward1
        all_ret1_models.append(all_ret1)
        all_ret2_models.append(all_ret2)

    # Combine ret1 and ret2 for each model
    combined_returns = [np.concatenate([ret1, ret2]) for ret1, ret2 in zip(all_ret1_models, all_ret2_models)]

    # Compute means and standard errors
    means = [np.mean(ret) for ret in combined_returns]
    stds = [np.std(ret) for ret in combined_returns]
    ses = [np.std(ret) / np.sqrt(len(ret)) for ret in combined_returns]

    # Bar plot
    x = np.arange(num_dir)
    fig, ax = plt.subplots(figsize=(5, 6))

    bars = ax.bar(x, means, yerr=ses, capsize=5, color='skyblue', edgecolor='black')

    # Add individual dots (ret1 in blue, ret2 in orange)
    for i in range(num_dir):
        jitter = 0.08
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret1_models[i])), all_ret1_models[i], '.', color='tab:blue',
                alpha=0.6, label='Return 1' if i == 0 else "")
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret2_models[i])), all_ret2_models[i], '.',
                color='tab:orange', alpha=0.6, label='Return 2' if i == 0 else "")

    t_stat, p_val = ttest_ind(combined_returns[0], combined_returns[1])
    print(t_stat, p_val)

    # plt.hlines(500, -0.5, 5.5, linestyle='dashed', color='k')
    # Final touches
    ax.set_xticks(x)
    ax.set_xticklabels(['with full gaze', 'with partial gaze'], rotation=45)
    ax.set_ylabel("repressing before cooperation")
    plt.tight_layout()
    plt.show()

def graph_rewards(dirs):
    # Example setup
    model_names = dirs
    num_dir = len(model_names)

    all_ret1_models = []
    all_ret2_models = []

    for model_dir in dirs:
        good = np.load(root_dir + model_dir + '/good_folds.npy', allow_pickle=True).item()
        if 'baseline_pause_13' in good:
            value = good.pop('baseline_pause_13', None)
        num_models = len(list(good.keys()))
        tot_models = 10 * (len(os.listdir(root_dir + model_dir)) // 10)
        eps = 2000 if model_dir != 'no_embed_full' else 1400

        all_ret1 = np.zeros(num_models)
        all_ret2 = np.zeros(num_models)
        for i, g in enumerate(good):
            filepath = root_dir + model_dir + '/evaluate/' + g + '_evaluate'
            reward1, reward2 = get_reward(filepath)

            all_ret1[i] = reward1
            all_ret2[i] = reward1
        all_ret1_models.append(all_ret1)
        all_ret2_models.append(all_ret2)

    # Combine ret1 and ret2 for each model
    combined_returns = [np.concatenate([ret1, ret2]) for ret1, ret2 in zip(all_ret1_models, all_ret2_models)]

    # Compute means and standard errors
    means = [np.mean(ret) for ret in combined_returns]
    stds = [np.std(ret) for ret in combined_returns]
    ses = [np.std(ret) / np.sqrt(len(ret)) for ret in combined_returns]

    # Bar plot
    x = np.arange(num_dir)
    fig, ax = plt.subplots(figsize=(5, 6))

    bars = ax.bar(x, means, yerr=ses, capsize=5, color='skyblue', edgecolor='black')

    # Add individual dots (ret1 in blue, ret2 in orange)
    for i in range(num_dir):
        jitter = 0.08
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret1_models[i])), all_ret1_models[i], '.', color='tab:blue',
                alpha=0.6, label='Return 1' if i == 0 else "")
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret2_models[i])), all_ret2_models[i], '.',
                color='tab:orange', alpha=0.6, label='Return 2' if i == 0 else "")

    t_stat, p_val = ttest_ind(combined_returns[0], combined_returns[1])
    print(t_stat, p_val)

    # plt.hlines(500, -0.5, 5.5, linestyle='dashed', color='k')
    # Final touches
    ax.set_xticks(x)
    ax.set_xticklabels(['with full gaze', 'with partial gaze'], rotation=45)
    ax.set_ylabel("cooperative successes")
    plt.tight_layout()
    plt.show()

def graph_interpress(dirs):
    # Example setup
    model_names = dirs
    num_dir = len(model_names)

    all_ret1_models = []

    for model_dir in dirs:
        good = np.load(root_dir + model_dir + '/good_folds.npy', allow_pickle=True).item()
        if 'baseline_pause_13' in good:
            value = good.pop('baseline_pause_13', None)
        num_models = len(list(good.keys()))
        tot_models = 10 * (len(os.listdir(root_dir + model_dir)) // 10)
        eps = 2000 if model_dir != 'no_embed_full' else 1400

        all_ret1 = np.zeros(num_models)
        for i, g in enumerate(good):
            filepath = root_dir + model_dir + '/evaluate/' + g + '_evaluate'
            interpress = get_interpress(filepath)
            all_ret1[i] = interpress
        all_ret1_models.append(all_ret1)

    # Combine ret1 and ret2 for each model
    combined_returns = all_ret1_models  # [np.concatenate([ret1, ret2]) for ret1, ret2 in zip(all_ret1_models, all_ret2_models)]

    # Compute means and standard errors
    means = [np.mean(ret) for ret in combined_returns]
    stds = [np.std(ret) for ret in combined_returns]
    ses = [np.std(ret) / np.sqrt(len(ret)) for ret in combined_returns]

    # Bar plot
    x = np.arange(num_dir)
    fig, ax = plt.subplots(figsize=(5, 6))

    bars = ax.bar(x, means, yerr=ses, capsize=5, color='skyblue', edgecolor='black')

    # Add individual dots (ret1 in blue, ret2 in orange)
    for i in range(num_dir):
        jitter = 0.08
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret1_models[i])), all_ret1_models[i], '.', color='tab:blue',
                alpha=0.6, label='Return 1' if i == 0 else "")
        # ax.plot(np.random.normal(x[i], jitter, size=len(all_ret2_models[i])), all_ret2_models[i], '.', color='tab:orange', alpha=0.6, label='Return 2' if i == 0 else "")

    t_stat, p_val = ttest_ind(combined_returns[0], combined_returns[1])
    print(t_stat, p_val)

    # plt.hlines(500, -0.5, 5.5, linestyle='dashed', color='k')
    # Final touches
    ax.set_xticks(x)
    ax.set_xticklabels(['with full gaze', 'with partial gaze'], rotation=45)
    ax.set_ylabel("all interpress interval")
    plt.tight_layout()
    plt.show()

def graph_ratid(dirs):
    # Example setup
    model_names = dirs
    num_dir = len(model_names)

    all_ret1_models = []

    for model_dir in dirs:
        good = np.load(root_dir + model_dir + '/good_folds.npy', allow_pickle=True).item()
        if 'baseline_pause_13' in good:
            value = good.pop('baseline_pause_13', None)
        num_models = len(list(good.keys()))
        tot_models = 10 * (len(os.listdir(root_dir + model_dir)) // 10)
        eps = 2000 if model_dir != 'no_embed_full' else 1400

        all_ret1 = np.zeros(num_models)
        for i, g in enumerate(good):
            filepath = root_dir + model_dir + '/evaluate/' + g + '_evaluate'
            interpress = get_ratid2(filepath)
            all_ret1[i] = interpress
        all_ret1_models.append(all_ret1)

    # Combine ret1 and ret2 for each model
    combined_returns = all_ret1_models  # [np.concatenate([ret1, ret2]) for ret1, ret2 in zip(all_ret1_models, all_ret2_models)]

    # Compute means and standard errors
    means = [np.mean(ret) for ret in combined_returns]
    stds = [np.std(ret) for ret in combined_returns]
    ses = [np.std(ret) / np.sqrt(len(ret)) for ret in combined_returns]

    # Bar plot
    x = np.arange(num_dir)
    fig, ax = plt.subplots(figsize=(5, 6))

    bars = ax.bar(x, means, yerr=ses, capsize=5, color='skyblue', edgecolor='black')

    # Add individual dots (ret1 in blue, ret2 in orange)
    for i in range(num_dir):
        jitter = 0.08
        ax.plot(np.random.normal(x[i], jitter, size=len(all_ret1_models[i])), all_ret1_models[i], '.', color='tab:blue',
                alpha=0.6, label='Return 1' if i == 0 else "")
        # ax.plot(np.random.normal(x[i], jitter, size=len(all_ret2_models[i])), all_ret2_models[i], '.', color='tab:orange', alpha=0.6, label='Return 2' if i == 0 else "")

    t_stat, p_val = ttest_ind(combined_returns[0], combined_returns[1])
    print(t_stat, p_val)

    # plt.hlines(500, -0.5, 5.5, linestyle='dashed', color='k')
    # Final touches
    plt.hlines(.5, -0.5, 1.5, 'k', linestyle='dashed')
    ax.set_xticks(x)
    ax.set_xticklabels(['with full gaze', 'with partial gaze'], rotation=45)
    ax.set_ylabel("p(more active rat initiating cooperation)")
    plt.tight_layout()
    plt.show()

def graph_actions(dirs):
    # Example setup
    model_names = dirs
    num_dir = len(model_names)

    all_ret1_models = []
    all_ret2_models = []

    for model_dir in dirs:
        good = np.load(root_dir + model_dir + '/good_folds.npy', allow_pickle=True).item()
        if 'baseline_pause_13' in good:
            value = good.pop('baseline_pause_13', None)
        num_models = len(list(good.keys()))
        tot_models = 10 * (len(os.listdir(root_dir + model_dir)) // 10)
        eps = 2000 if model_dir != 'no_embed_full' else 1400

        all_ret1 = np.zeros((num_models, 3))
        all_ret2 = np.zeros((num_models, 3))
        for i, g in enumerate(good):
            filepath = root_dir + model_dir + '/evaluate/' + g + '_evaluate'
            mv_act, gz_act = get_actions(filepath)
            all_ret1[i] = mv_act
            all_ret2[i] = gz_act
        all_ret1_models.append(all_ret1)
        all_ret2_models.append(all_ret2)

    datas = [all_ret1_models, all_ret2_models]
    for data in datas:
        # Aggregate counts of actions for each model
        model1_action_counts = np.mean(data[0], axis=0)
        model2_action_counts = np.mean(data[1], axis=0)

        # Plotting
        actions = ['Action 0', 'Action 1', 'Action 2']
        x = np.arange(len(actions))

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Model 1 histogram
        axs[0].bar(x, model1_action_counts, color='skyblue')
        axs[0].set_title('Full Gaze Action Frequency')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(actions)
        axs[0].set_ylabel('Average Frequency')

        # Model 2 histogram
        axs[1].bar(x, model2_action_counts, color='orange')
        axs[1].set_title('Partial Gaze Action Frequency')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(actions)

        plt.tight_layout()
        plt.show()


