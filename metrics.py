import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


total_eps = 1000

def get_reward(filepath):
    rewards = np.load(f'{filepath}/all_rewards.pkl', allow_pickle=True)
    
    total_reward1 = 0
    total_reward2 = 0
    for eps in rewards.keys():
        total_reward1 += (len(rewards[eps][0]))
        total_reward2 += len(rewards[eps][1])
    avg_reward1 = total_reward1 / total_eps
    avg_reward2 = total_reward2 / total_eps
    return avg_reward1, avg_reward2

def get_interpress(filepath):
    pulls = np.load(filepath + '/all_pulls.pkl', allow_pickle=True)
    rewards = np.load(filepath + '/all_rewards.pkl', allow_pickle=True)
    interpress = []
    for eps in rewards.keys():
        pull1 = pulls[eps][0]
        pull2 = pulls[eps][1]
        for r in rewards[eps][0]:
            pp1 = [x for x in pull1 if x < r]
            pp2 = [x for x in pull2 if x < r]
            all_p = np.concatenate((pp1, pp2))

            try:
                max_p = np.max(all_p)

                if max_p in pp1:

                    intt = max_p - np.min(pp2)
                    if intt > 7:
                        inter = max_p * np.ones(len(pp2)) - (pp2)
                        inter = [x for x in inter if x <= 7]
                        inter = np.max(inter)
                    interpress.append(intt)
                else:
                    intt = max_p - np.min(pp1)
                    if intt > 7:
                        inter = max_p * np.ones(len(pp1)) - (pp1)
                        inter = [x for x in inter if x <= 7]
                        inter = np.max(inter)
                    interpress.append(intt)
                pull1 = [x for x in pull1 if x not in pp1]
                pull2 = [x for x in pull2 if x not in pp2]
            except:
                continue
            # find max
            # find min in other
            # subtract...

    return np.mean(interpress)

def get_repress(filepath):
    pulls = np.load(filepath + '/all_pulls.pkl', allow_pickle=True)
    rewards = np.load(filepath + '/all_rewards.pkl', allow_pickle=True)
    repress1 = 0
    repress2 = 0
    for eps in rewards.keys():
        pull1 = pulls[eps][0]
        pull2 = pulls[eps][1]
        for r in rewards[eps][0]:
            pp1 = [x for x in pull1 if x  < r]
            pp2 = [x for x in pull2 if x  < r]
            repress1 += (len(pp1) - 1)
            repress2 += (len(pp2) - 1)
            pull1 = [x for x in pull1 if x not in pp1]
            pull2 = [x for x in pull2 if x not in pp2]
    return repress1 / 1000, repress2 / 1000

def get_rat_id(filepath):
    first = np.load(f'{filepath}/first_pulls.pkl', allow_pickle=True)
    coop = np.load(f'{filepath}/coop_pulls.pkl', allow_pickle=True)
    pulls = np.load(f'{filepath}/all_pulls.pkl', allow_pickle=True)

    # want rat id of all first pulls and all coop pulls
    first_ids = []
    coop_ids = []
    for eps in first.keys():
        rat0_pulls = pulls[eps][0]
        rat1_pulls = pulls[eps][1]
        for c in coop[eps]:
            if c in rat0_pulls:
                coop_ids.append(0)
                first_ids.append(1)
            elif c in rat1_pulls:
                coop_ids.append(1)
                first_ids.append(0)
            else:
                print('no id')
        # TAKING THIS OUT RIGHT NOW BECAUSE OUR "FIRST PULL" IS DIFF THAN THEIRS...
        # BECAUSE OUR LEVER DOESN'T GO IN THE COOP THRESHOLD AMT OF TIME AFTER IT GOES OUT
        # for f in first[eps]:
        #     if f in rat0_pulls:
        #         first_ids.append(0)
        #     elif f in rat1_pulls:
        #         first_ids.append(1)
        #     else:
        #         print('no id')
    return np.mean(first_ids), np.mean(coop_ids)


def get_ratid2(filepath):
    pulls = np.load(filepath + '/all_pulls.pkl', allow_pickle=True)
    rewards = np.load(filepath + '/all_rewards.pkl', allow_pickle=True)
    interpress = []
    for eps in rewards.keys():
        pull1 = pulls[eps][0]
        pull2 = pulls[eps][1]
        for r in rewards[eps][0]:
            pp1 = [x for x in pull1 if x < r]
            pp2 = [x for x in pull2 if x < r]
            all_p = np.concatenate((pp1, pp2))

            try:
                max_p = np.max(all_p)

                if max_p in pp1:

                    interpress.append(1)
                else:
                    interpress.append(0)
                pull1 = [x for x in pull1 if x not in pp1]
                pull2 = [x for x in pull2 if x not in pp2]
            except:
                continue
            # find max
            # find min in other
            # subtract...
    rat_id = np.mean(interpress)
    return rat_id if rat_id > 0.5 else 1 - rat_id

def get_gaze(filepath):
    actions = np.load(f'{filepath}/all_actions.pkl', allow_pickle=True)
    all_gaze_actions0 = []
    all_gaze_actions1 = []
    for eps in actions.keys():
        rat0_actions = actions[eps][0]
        rat1_actions = actions[eps][1]
        for step in range(len(rat0_actions)):
            gaze0 = rat0_actions[step][1]
            gaze1 = rat1_actions[step][1]
            all_gaze_actions0.append(gaze0)
            all_gaze_actions1.append(gaze1)
    return np.sum(all_gaze_actions0) / total_eps, np.sum(all_gaze_actions1) / total_eps

def get_mutual_gaze(filepath):
    actions = np.load(f'{filepath}/all_actions.pkl', allow_pickle=True)
    total_mutual_gaze = 0
    for eps in actions.keys():
        rat0_actions = actions[eps][0]
        rat1_actions = actions[eps][1]
        for step in range(len(rat0_actions)):
            gaze0 = rat0_actions[step][1]
            gaze1 = rat1_actions[step][1]
            if gaze0 == 1 and gaze1 == 1:
                total_mutual_gaze += 0
    return total_mutual_gaze / total_eps

def get_actions(filepath):
    actions = np.load(filepath + '/all_actions.pkl', allow_pickle=True)

    mv_act = []
    gz_act = []
    for eps in actions.keys():

        act1 = actions[eps][0]
        act2 = actions[eps][1]
        for i, a1 in enumerate(act1):
            if a1[1] == 0:
                mv_act.append(a1[0])
            else:
                gz_act.append(a1[0])

            if act2[i][1] == 0:
                mv_act.append(act2[i][0])
            else:
                gz_act.append(act2[i][0])
    mv_count = Counter(mv_act)
    gz_count = Counter(gz_act)

    mv_act = np.array([mv_count[0], mv_count[1], mv_count[2]])
    gz_act = np.array([gz_count[0], gz_count[1], gz_count[2]])
    return mv_act / 1000, gz_act / 1000


# old graphing methods don't actually know if they work...
# def graph_reward(model):
#     x = []
#     r = []
#     for e in epochs:
#         filepath = savedir + model + e
#         x.append(int(e.split('_')[-1]))
#         r.append(get_reward(filepath))
#     plt.plot(x, r, '.-')
#     plt.xlabel('% of training')
#     plt.ylabel('average episode reward')
#     plt.show()
#
# def graph_inter_press(model):
#     x = []
#     r = []
#     for e in epochs:
#         filepath = savedir + model + e
#         x.append(int(e.split('_')[-1]))
#         r.append(get_inter_press(filepath))
#     plt.plot(x, r, '.-')
#     plt.xlabel('% of training')
#     plt.ylabel('average time between first press and cooperative press')
#     plt.show()
#
# def graph_repress(model):
#     x = []
#     r = []
#     for e in epochs:
#         filepath = savedir + model + e
#         x.append(int(e.split('_')[-1]))
#         r.append(get_repress(filepath))
#     plt.plot(x, r, '.-')
#     plt.xlabel('% of training')
#     plt.ylabel('average amount of repress')
#     plt.show()
#
# def graph_rat_id(model):
#     x = []
#     f = []
#     c = []
#     for e in epochs:
#         filepath = savedir + model + e
#         x.append(int(e.split('_')[-1]))
#         first, coop = get_rat_id(filepath)
#         f.append(first)
#         c.append(coop)
#     plt.plot(x, f, '.-', label='first')
#     plt.plot(x, c, '.-', label='coop')
#     plt.hlines(0.5, 0, 100, linestyle='dashed', color='tab:grey')
#     plt.legend()
#     plt.xlabel('% of training')
#     plt.ylabel('rat id')
#     plt.show()
#
# def graph_gaze(model):
#     x = []
#     z = []
#     o = []
#     for e in epochs:
#         filepath = savedir + model + e
#         x.append(int(e.split('_')[-1]))
#         zero, one = get_gaze(filepath)
#         z.append(zero)
#         o.append(one)
#     plt.plot(x, z, '.-', label='rat zero')
#     plt.plot(x, o, '.-', label='rat one')
#     plt.legend()
#     plt.xlabel('% of training')
#     plt.ylabel('number of gazes')
#     plt.show()
#
# def graph_mutual_gaze(model):
#     x = []
#     r = []
#     for e in epochs:
#         filepath = savedir + model + e
#         x.append(int(e.split('_')[-1]))
#         r.append(get_mutual_gaze(filepath))
#     plt.plot(x, r, '.-')
#     plt.xlabel('% of training')
#     plt.ylabel('average number of social gaze')
#     plt.show()