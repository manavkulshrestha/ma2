import numpy as np
from sim.scenario import spline_scenario


np.random.seed(42)


def main():
    seed = np.random.randint(1, 10000)
    np.random.seed(seed)

    print(f'{seed=}')
    for _ in range(10000):
        run_data = spline_scenario(scene_len=1000, render=True, human_rng=(1,2), robot_rng=(1,2), display_spline_idx=None)


if __name__ == '__main__':
    main()