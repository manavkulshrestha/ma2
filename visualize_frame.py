import time
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from sim.make_env import make_env
from sim.utility import load_pkl, sliding


def visualize(h_state, r_state):
    env = make_env('simple_herding', benchmark=False,
                   num_humans=len(h_state), num_robots=len(r_state), num_goals=0)
    
    obs_n = env.reset()

    for h, state in zip(env.world.humans, h_state):
        h.state.p_pos = state[:2]

    for r, state in zip(env.world.robots, r_state):
        r.state.p_pos = state[:2]

    env.step(np.zeros([len(h_state)+len(r_state), 2]))
    env.render()

def visual_comparision(data, t0, t1):
    dt0, dt1 = data[t0], data[t1]

    visualize(dt0['h_state'], dt0['r_state'])
    visualize(dt1['h_state'], dt1['r_state'])
    print(dt0['r_actions'])

    human_diffs = np.linalg.norm(dt1['h_state'][:,:2]-dt0['h_state'][:,:2], axis=1)
    robot_diffs = np.linalg.norm(dt1['r_state'][:,:2]-dt0['r_state'][:,:2], axis=1)
    print(f'human_diffs = {human_diffs}')
    print(f'robot_diffs = {robot_diffs}')
    time.sleep(5000)

def diff_analysis(data, stride=1):
    ts = data[::stride]

    human_diff_means = []
    robot_diff_means = []

    for ti, tj in sliding(ts, 2):
        human_diffs = np.linalg.norm(ti['h_state'][:,:2]-tj['h_state'][:,:2], axis=1)
        robot_diffs = np.linalg.norm(ti['r_state'][:,:2]-tj['r_state'][:,:2], axis=1)

        human_diff_means.append(human_diffs.mean())
        robot_diff_means.append(robot_diffs.mean())

    # plt.figure(figsize=(10, 6))
    # plt.hist(robot_diff_means, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    # plt.title('Histogram of Data')
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.show()

    print(f'human diff mean of means {np.mean(human_diff_means)}')
    print(f'robot diff mean of means {np.mean(robot_diff_means)}')


def main():
    old = load_pkl('data/spline_i-5673/24-01-22-14254165')['timeseries'] # old
    # new = load_pkl('data/spline_i-5700/24-02-02-13280766')['timeseries'] # new
    # new = load_pkl('data/spline_i-5700/24-02-02-13283522')['timeseries'] # new

    visual_comparision(old, 250, 251)
    diff_analysis(old, stride=1)

if __name__ == '__main__':
    main()