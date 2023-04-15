import os
import copy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from Pathfinder import my_mkdir
import vanet.env_params as p
import seaborn as sns

sns.set()

plt.switch_backend('cairo')

plot_params = {
    'lines.linewidth': 1.3,
    'figure.titlesize': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    # 'font.sans-serif': 'Times New Roman'
}
plt.rcParams.update(plot_params)

color_list = ['tab:purple', 'tab:brown', 'tab:green', 'tab:orange', 'tab:blue']
moving_average_step = 20

JOINT_BASELINES, BASELINE_RES = [], []
with open('./Outcome/baselines/outcome.txt', 'r') as file:
    for line in file.readlines():
        comb_alg_name, test_res = line.split(':')
        JOINT_BASELINES.append(comb_alg_name)
        BASELINE_RES.append(list(map(float, test_res.split(' '))))


# 平滑滤波
def np_move_avg(a, mode='valid'):
    return np.convolve(a, np.ones((moving_average_step,)) / moving_average_step, mode=mode)


def draw_performance(metric_index):
    performance_metric = p.METRICS[metric_index]
    performance_metric_path_str = '-'.join(performance_metric.split(' '))
    baseline_res_list = [BASELINE_RES[i][metric_index] for i in range(len(JOINT_BASELINES))]

    def changex1(temp, position):
        return int(temp * 10)

    if not Path('./Graph').is_dir():
        os.system('mkdir Graph')

    test_group_List = os.listdir('./Outcome')
    all_algs_metric_dict = {}
    for env_set in test_group_List:
        if env_set == 'baselines':
            continue

        min_all_len = float('inf')
        if not Path(f'./Graph/{env_set}/Summary').is_dir():
            my_mkdir(os.path.join('/Graph', f'{env_set}/Summary'))
        for drl_alg in os.listdir(os.path.join('./Outcome', env_set)):
            plot_len = float('inf')
            avg_res_list, min_res_list, max_res_list = [], [], []
            res_outcome_list = [avg_res_list, max_res_list, min_res_list]
            if not Path(f'./Graph/{env_set}/{drl_alg}/single_summary').is_dir():
                my_mkdir(os.path.join('/Graph', f'{env_set}/{drl_alg}/single_summary'))
            seed_list = os.listdir(f'./Outcome/{env_set}/{drl_alg}')
            for seed in seed_list:
                if not Path(f'./Graph/{env_set}/{drl_alg}/{seed}/').is_dir():
                    my_mkdir(f'/Graph/{env_set}/{drl_alg}/{seed}')

                # performance
                res_list = []

                with open(f'./Outcome/{env_set}/{drl_alg}/{seed}/{performance_metric_path_str}.txt', 'r') as rfile:
                    for line in rfile.readlines():
                        new_res = float((line.split(':')[1]).split('\n')[0])
                        res_list.append(new_res)
                        if len(res_outcome_list[0]) >= len(res_list):
                            res_outcome_list[0][len(res_list) - 1] += new_res
                            if new_res > res_outcome_list[1][len(res_list) - 1]:
                                res_outcome_list[1][len(res_list) - 1] = new_res
                            if new_res < res_outcome_list[2][len(res_list) - 1]:
                                res_outcome_list[2][len(res_list) - 1] = new_res
                        for outcome_tmp_list in res_outcome_list:
                            if len(outcome_tmp_list) < len(res_list):
                                outcome_tmp_list.append(new_res)

                if len(res_list) < plot_len:
                    plot_len = len(res_list)

                title = f'{env_set}-{drl_alg}-{seed}'

                plt.clf()
                plt.figure(2, figsize=(14, 7))
                plt.title(title)
                plt.gca().xaxis.set_major_formatter(FuncFormatter(changex1))
                plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                plt.plot(np.arange(len(res_list)) + 1, res_list, label=drl_alg, color='green')
                plt.xlabel('Training episodes')
                plt.ylabel(performance_metric)
                plt.grid(True, 'major', 'y')
                plt.legend(loc='best')
                plt.savefig(f'./Graph/{env_set}/{drl_alg}/{seed}/{performance_metric_path_str}.png')

                plt.clf()
                plt.close()

            avg_res_list = [x / len(seed_list) for x in avg_res_list]

            if min_all_len > plot_len:
                min_all_len = plot_len

            all_algs_metric_dict[drl_alg] = (
                copy.deepcopy(avg_res_list[0:plot_len]), copy.deepcopy(min_res_list[0:plot_len]),
                copy.deepcopy(max_res_list[0:plot_len]))
            plt.figure(6, figsize=(14, 7))
            plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            plt.gca().xaxis.set_major_formatter(FuncFormatter(changex1))
            plt.plot(np.arange(plot_len) + 1, avg_res_list[0:plot_len])
            plt.fill_between(np.arange(plot_len) + 1, min_res_list[0:plot_len], max_res_list[0:plot_len], alpha=0.3)
            plt.title(f'{env_set}-{drl_alg}-{seed}-Summary ({performance_metric})')
            plt.xlabel('Training Episodes')
            plt.ylabel(performance_metric)
            plt.grid(True, 'major', 'y')
            plt.savefig(f'./Graph/{env_set}/{drl_alg}/single_summary/{performance_metric_path_str}.png')

            plt.clf()

        plt.clf()
        plt.figure(8, figsize=(14, 7))
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(changex1))
        count = 0
        for key, results in all_algs_metric_dict.items():
            plt.plot(np.arange(min_all_len - moving_average_step + 1) + 1, np_move_avg(results[0][0:min_all_len]), color=color_list[count], label=key)
            plt.fill_between(np.arange(min_all_len - moving_average_step + 1) + 1, np_move_avg(results[1][0:min_all_len]), np_move_avg(results[2][0:min_all_len]),
                             alpha=0.3, facecolor=color_list[count])
            count += 1

        # 绘制baselines
        for i, res_i in enumerate(baseline_res_list):
            plt.plot(np.arange(min_all_len) + 1, [res_i for _ in range(min_all_len)], color=color_list[count + i],
                     label=f'{JOINT_BASELINES[i]}')

        plt.title(f'{env_set}-{seed}-Summary ({performance_metric})')
        plt.xlabel('Training Episodes')
        plt.ylabel(performance_metric)
        plt.grid(True, 'major', 'y')
        plt.legend()
        plt.savefig(f'./Graph/{env_set}/Summary/all_algs_{performance_metric_path_str}.png')

        plt.close()


def draw_loss(node_type):

    def changex2(temp, position):
        return int(temp)

    test_group_List = os.listdir('./Outcome')
    for env_set in test_group_List:
        if env_set == 'baselines':
            continue

        for drl_alg in os.listdir(os.path.join('./Outcome', env_set)):
            plot_len = float('inf')
            avg_loss_list, max_loss_list, min_loss_list = [], [], []
            loss_outcome_list = [avg_loss_list, max_loss_list, min_loss_list]
            seed_list = os.listdir(f'./Outcome/{env_set}/{drl_alg}')
            for seed in seed_list:
                agent_loss_list = []

                with open(f'./Outcome/{env_set}/{drl_alg}/{seed}/{node_type}/agent_loss.txt', 'r') as alossfile:
                    for line in alossfile.readlines():
                        aloss = float(line.split('\n')[0])
                        agent_loss_list.append(aloss)
                        if len(loss_outcome_list[0]) >= len(agent_loss_list):
                            loss_outcome_list[0][len(agent_loss_list) - 1] += aloss
                            if aloss > loss_outcome_list[1][len(agent_loss_list) - 1]:
                                loss_outcome_list[1][len(agent_loss_list) - 1] = aloss
                            if aloss < loss_outcome_list[2][len(agent_loss_list) - 1]:
                                loss_outcome_list[2][len(agent_loss_list) - 1] = aloss
                        for outcome_tmp_list in loss_outcome_list:
                            if len(outcome_tmp_list) < len(agent_loss_list):
                                outcome_tmp_list.append(aloss)

                    # with open(f'./Outcome/{env_set}/{drl_alg}/{seed}/c_loss.txt', 'r') as clossfile:
                    #     for line in clossfile.readlines():
                    #         closs = float(line.split('\n')[0])
                    #         closs_list.append(closs)

                if len(agent_loss_list) < plot_len:
                    plot_len = len(agent_loss_list)

                title = f'{env_set}-{drl_alg}-{seed}-Agent Training Loss'

                plt.clf()
                plt.figure(4, figsize=(14, 7))
                plt.title(title)
                plt.gca().xaxis.set_major_formatter(FuncFormatter(changex2))
                plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                plt.plot(np.arange(len(agent_loss_list)) + 1, agent_loss_list, label=drl_alg, color='red')
                plt.xlabel('Training steps')
                plt.ylabel(f'{node_type} Agent Loss')
                plt.grid(True, 'major', 'y')
                plt.legend(loc='best')
                plt.savefig(f'./Graph/{env_set}/{drl_alg}/{seed}/{node_type}_agent_loss.png')

            avg_loss_list = [x / len(seed_list) for x in avg_loss_list]

            plt.clf()
            plt.figure(8, figsize=(14, 7))
            plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            plt.gca().xaxis.set_major_formatter(FuncFormatter(changex2))
            plt.plot(np.arange(plot_len - moving_average_step + 1) + 1, np_move_avg(avg_loss_list[0:plot_len]), color=color_list[0], label=f'{node_type}')
            plt.fill_between(np.arange(plot_len - moving_average_step + 1) + 1,
                             np_move_avg(max_loss_list[0:plot_len]), np_move_avg(min_loss_list[0:plot_len]),
                             alpha=0.3, facecolor=color_list[0])

            plt.title(f'{env_set}-{seed}-Summary (Agent Training Loss)')
            plt.xlabel('Training steps')
            plt.ylabel(f'{node_type} Agent Loss')
            plt.grid(True, 'major', 'y')
            plt.legend()
            plt.savefig(f'./Graph/{env_set}/Summary/{node_type}_loss.png')

        plt.close()


if __name__ == '__main__':
    for metric_i in range(len(p.METRICS)):
        draw_performance(metric_i)
    for node_type in ['RSU', 'Vehicle']:
        draw_loss(node_type)
