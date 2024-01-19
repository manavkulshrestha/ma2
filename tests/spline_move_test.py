import numpy as np
from ..scenario import spline_scenario


def main():
    seed = np.random.randint(1, 10000)
    np.random.seed(seed)

    print('{seed=}')
    for _ in range(10000):
        run_data = spline_scenario(scene_len=1000, render=True, human_rng=(0,2), robot_rng=(0,1))


if __name__ == '__main__':
    main()