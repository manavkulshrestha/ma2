from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt

from sim.scenario import spline_scenario
from data.datacollection import save_data
from sim.utility import load_pkl, plot_xy


def main():
    seed = np.random.randint(1, 10000) # 4914 or 
    seed_everything(seed)

    N = 10000
    record = True
    test = False

    sub_dir = Path('data')/f'spline_i-{seed}'
    if record:
        sub_dir.mkdir(exist_ok=False)

    norms = []

    print(f'{seed=}')
    for i in tqdm(range(N)):
        run_data = spline_scenario(scene_len=1000, human_rng=(7, 10), robot_rng=(7, 10), render=False, verbose=False, spline_degree=1)

        filename = f'{i:0{len(str(N))}d}.pkl'
        if test:
            collected_state = np.array([[*t['h_state'], *t['r_state']] for t in run_data[0]])
            recorded_state = np.array([[*t['h_state'], *t['r_state']] for t in load_pkl(sub_dir/filename)['timeseries']])

            norms.append(np.linalg.norm(collected_state-recorded_state))

        saved_data = save_data(*run_data, sub_dir=sub_dir, name=filename, write=record)

    # print(norms)
    # print(sum(norms))
    # print(sum(norms)/N)


        # plot_xy(np.vstack([t['r_actions'] for t in run_data[0]]))
        # plot_xy(np.vstack([t['r_actions'] for t in saved_data['timeseries']]))

        # print('', end='')


if __name__ == '__main__':
    main()