from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt

from sim.make_env import make_env
from sim.scenario import spline_scenario
from data.datacollection import save_data
from sim.utility import load_pkl, plot_xy


def main():
    seed = 1574 or np.random.randint(1, 10000) # 4914 or 
    seed_everything(seed)

    num_scenes, scene_len = 5000, 500
    render = False
    record = not render

    test = False

    sub_dir = Path('data')/f'spline_i-{seed}'
    if record:
        sub_dir.mkdir(exist_ok=False)

    norms = []

    print(f'{seed=}')
    for i in tqdm(range(num_scenes)):
        num_humans, num_robots, num_goals = np.random.randint(*np.vstack([(7, 10), (7, 10), (0, 1)]).T)
        env = make_env('simple_herding', benchmark=False,
                       num_humans=num_humans, num_robots=num_robots, num_goals=num_goals, action_noise=0)
    # print(num_robots, num_humans, num_goals)
        obs_n = env.reset()

        if i % 10 == 0:
            print('tested', i)

    # print(norms)
    # print(sum(norms))
    # print(sum(norms)/N)


        # plot_xy(np.vstack([t['r_actions'] for t in run_data[0]]))
        # plot_xy(np.vstack([t['r_actions'] for t in saved_data['timeseries']]))

        # print('', end='')


if __name__ == '__main__':
    main()