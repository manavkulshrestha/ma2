from itertools import islice
import time
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from data.datasets import all_paths

from sim.make_env import make_env
from sim.utility import chunked, load_pkl, plot_xy, sliding


def old_comparision():
    old = load_pkl('data/spline_i-5673/24-01-22-14254165')['timeseries'] # old
    # new = load_pkl('data/spline_i-5700/24-02-02-13280766')['timeseries'] # new
    # new = load_pkl('data/spline_i-5700/24-02-02-13283522')['timeseries'] # new

    visual_comparision(old, 250, 251)
    diff_analysis(old, stride=1)

def visualize(file_path):
    file = load_pkl(file_path)
    timeseries = file['timeseries']

    env = make_env('simple_herding', benchmark=False,
                   num_humans=file['num_humans'], num_robots=file['num_robots'], num_goals=0)
    
    obs_n = env.reset()

    for t in timeseries:
        h_state, r_state = t['h_state'], t['r_state']

        for h, state in zip(env.world.humans, h_state):
            h.state.p_pos = state[:2]

        for r, state in zip(env.world.robots, r_state):
            r.state.p_pos = state[:2]

        env.step(np.zeros([len(h_state)+len(r_state), 2]))
        env.render()

def resimulate_and_visualize(file_path):
    file = load_pkl(file_path)
    timeseries = file['timeseries']

    # initial positions
    env_r = make_env('simple_herding', benchmark=False,
                     num_humans=file['num_humans'], num_robots=file['num_robots'], num_goals=0)
    env_v = make_env('simple_herding', benchmark=False,
                     num_humans=file['num_humans'], num_robots=file['num_robots'], num_goals=0)
    
    # set initial position to same as recorded
    obs_n = env_v.reset()
    obs_n = env_r.reset()


    (hpos, hvel), (rpos, rvel) = [np.hsplit(timeseries[0][x], 2) for x in ('h_state', 'r_state')]
    for h, pos, vel in zip(env_r.world.humans, hpos, hvel):
        h.state.p_pos = pos
        h.state.p_vel = vel
    for r, pos, vel in zip(env_r.world.robots, rpos, rvel):
        r.state.p_pos = pos
        r.state.p_vel = vel

    for t in timeseries:
        (hpos, hvel), (rpos, rvel) = [np.hsplit(t[x], 2) for x in ('h_state', 'r_state')]
        act_n = np.concatenate([t['h_actions'], t['r_actions']])

        # compare vel and pos with recorded
        hpos_a, hvel_a, rpos_a, rvel_a = [], [], [], []
        for h in env_r.world.humans:
            hpos_a.append(h.state.p_pos)
            hvel_a.append(h.state.p_vel)
        for r in env_r.world.robots:
            rpos_a.append(r.state.p_pos)
            rvel_a.append(r.state.p_vel)

        hpos_a, hvel_a, rpos_a, rvel_a = [np.array(x) for x in (hpos_a, hvel_a, rpos_a, rvel_a)]
        
        # visualization looks fine, these values are not good
        hpe = np.linalg.norm(hpos-hpos_a, axis=1).mean()
        hve = np.linalg.norm(hvel-hvel_a, axis=1).mean()
        rpe = np.linalg.norm(rpos-rpos_a, axis=1).mean()
        rve = np.linalg.norm(rvel-rvel_a, axis=1).mean()

        print(f'{hpe=}, {hve=}, {rpe=}, {rve=}')
        tol = 1e-12
        if (np.array([hpe, hve, rpe, rve]) > 2).any():
            print()
        # assert (np.array([hpe, hve, rpe, rve]) < tol).all() 

        env_r.step(act_n)
        env_r.render()

        for h, pos, vel in zip(env_v.world.humans, hpos, hvel):
            h.state.p_pos = pos
            # h.state.p_vel = vel

        for r, pos, vel in zip(env_v.world.robots, rpos, rvel):
            r.state.p_pos = pos
            # r.state.p_vel = vel

        env_v.step(np.zeros([file['num_humans']+file['num_robots'], 2]))
        env_v.render()
    

def resimulate(file_path):
    file = load_pkl(file_path)
    timeseries = file['timeseries']

    # initial positions
    env = make_env('simple_herding', benchmark=False,
                   num_humans=file['num_humans'], num_robots=file['num_robots'], num_goals=0)
    
    # set initial position to same as recorded
    obs_n = env.reset()
    (hpos, hvel), (rpos, rvel) = [np.hsplit(timeseries[0][x], 2) for x in ('h_state', 'r_state')]
    for h, pos, vel in zip(env.world.humans, hpos, hvel):
        h.state.p_pos = pos
        h.state.p_vel = vel
    for r, pos, vel in zip(env.world.robots, rpos, rvel):
        r.state.p_pos = pos
        r.state.p_vel = vel

    h_state_a, r_state_a = [], []
    h_state_r, r_state_r = [], []

    for t in timeseries:
        (hpos, hvel), (rpos, rvel) = [np.hsplit(t[x], 2) for x in ('h_state', 'r_state')]
        act_n = np.concatenate([t['h_actions'], t['r_actions']])

        # compare vel and pos with recorded
        hpos_a, hvel_a, rpos_a, rvel_a = [], [], [], []
        for h in env.world.humans:
            hpos_a.append(h.state.p_pos)
            hvel_a.append(h.state.p_vel)

            h_state_a.append(np.concatenate([h.state.p_pos, h.state.p_vel]))

        for r in env.world.robots:
            rpos_a.append(r.state.p_pos)
            rvel_a.append(r.state.p_vel)

            r_state_a.append(np.concatenate([r.state.p_pos, r.state.p_vel]))

        h_state_r.append(t['h_state'])
        r_state_r.append(t['r_state'])

        hpos_a, hvel_a, rpos_a, rvel_a = [np.array(x) for x in (hpos_a, hvel_a, rpos_a, rvel_a)]
        
        hpe = np.linalg.norm(hpos-hpos_a)
        hve = np.linalg.norm(hvel-hvel_a)
        rpe = np.linalg.norm(rpos-rpos_a)
        rve = np.linalg.norm(rvel-rvel_a)

        print(f'{hpe=}, {hve=}, {rpe=}, {rve=}')
        tol = 1e-12
        # assert (np.array([hpe, hve, rpe, rve]) < tol).all() 

        env.step(act_n)
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

def acc_distr(file_path, skip_val):
    # path = Path('data/spline_i-106/')

    # start, end = 0, 800
    # files = list(filter(lambda p: p.is_file(), sorted(path.iterdir())[start:end]))
    # timeseries = load_pkl(files[1])['timeseries']
    
    timeseries = load_pkl(file_path)['timeseries']

    acts = []
    for i, ts in islice(enumerate(timeseries), 10+skip_val, None, skip_val): # skip first 10 frames
        sum_action = np.array([s['r_actions'] for s in timeseries[i-skip_val:i]]).sum(axis=0)

        acts.append(sum_action)

    acts = np.vstack(acts)
    print(len(acts))
    print(len(set(acts.flatten())))
    print(acts)

    plot_xy(acts)

def plot_acts(seed, see_every=100):
    paths = all_paths(seed)
    print(len(paths))

    for i, p in islice(enumerate(paths), None, None, see_every):
        check = np.vstack([t['r_actions'] for t in load_pkl(p)['timeseries']])
        print(i)
        plot_xy(check, autokill=0.5)
        # visualize(p)

def plot_trajectory(file, block=True):
    # Load the data
    ts = load_pkl(file)['timeseries']
    r_trajs = np.array([s['r_state'][:, :2] for s in ts])
    h_trajs = np.array([s['h_state'][:, :2] for s in ts])

    # Create subplots with 2 rows and 1 column
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Plot Robot Trajectories
    axs[0].set_aspect('equal', adjustable="datalim")

    axs[0].set_title('Robot Trajectories')
    for i in range(r_trajs.shape[1]):
        axs[0].plot(r_trajs[:, i, 0], r_trajs[:, i, 1])
    axs[0].set_xlabel('X coordinate')
    axs[0].set_ylabel('Y coordinate')
    # axs[0].set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal

    # Plot Human Trajectories
    axs[1].set_aspect('equal', adjustable="datalim")
    
    axs[1].set_title('Human Trajectories')
    for i in range(h_trajs.shape[1]):
        axs[1].plot(h_trajs[:, i, 0], h_trajs[:, i, 1])
    axs[1].set_xlabel('X coordinate')
    axs[1].set_ylabel('Y coordinate')
    # axs[1].set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal


    # axs[0].set_xlim(-1, 1)
    # axs[0].set_ylim(-1, 1)
    # axs[1].set_xlim(-1, 1)
    # axs[1].set_ylim(-1, 1)

    # Adjust layout
    # plt.tight_layout()

    # Show the plot without blocking
    plt.show(block=block)
    # plt.pause(0.001)
    # time.sleep(0.001)


def plot_actions(file, block=True):
    ts = load_pkl(file)['timeseries']
    acts = np.array([s['r_actions'] for s in ts])

    # Create a time array for the x-axis
    time = np.arange(acts.shape[0])

    # Create subplots for x and y accelerations
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot x accelerations
    for agent in range(acts.shape[1]):
        axs[0].plot(time, acts[:, agent, 0], label=f'Robot {agent+1}')

    axs[0].set_title('X Accelerations')
    axs[0].set_xlabel('Timestep')
    axs[0].set_ylabel('Acceleration')
    axs[0].legend()

    # Plot y accelerations
    for agent in range(acts.shape[1]):
        axs[1].plot(time, acts[:, agent, 1], label=f'Robot {agent+1}')

    axs[1].set_title('Y Accelerations')
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('Acceleration')
    axs[1].legend()

    plt.tight_layout()
    plt.show(block=block)


def main():
#     paths = all_paths(9613)
#     t = []

#     for p in paths:
#         acts = np.array([t['r_actions'] for t in load_pkl(p)['timeseries']])
#         # print(acts.min(), acts.max())
#         t.append(acts.min())
#         t.append(acts.max())

#     t = np.array(t)
#     print(t.min(), t.max())
        

    # paths = all_paths(6635)
    # visualisze(paths[900])

    # paths = all_paths(7644)
    # visualize(paths[900])

    paths = all_paths(652)
    plot_trajectory(paths[10], block=False)
    plot_actions(paths[10])
    # visualize(paths[0])



if __name__ == '__main__':
    main()