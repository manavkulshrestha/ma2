from pathlib import Path
import numpy as np
from tqdm import tqdm

from sim.scenario import linear_scenario, spline_scenario
from data.datacollection import save_data


def main():
    seed = np.random.randint(1, 10000)
    np.random.seed(seed)

    N = 10000
    record = True

    if record:
        sub_dir = Path('data')/f'spline_i-{seed}'
        sub_dir.mkdir(exist_ok=False)

    print(f'{seed=}')
    for _ in tqdm(range(N)):
        run_data = spline_scenario(scene_len=1000, human_rng=(7, 10), robot_rng=(7, 10), render=False, verbose=False)
        if record:
            save_data(*run_data, sub_dir=sub_dir)


if __name__ == '__main__':
    main()