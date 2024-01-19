import numpy as np
import time

from make_env import *

from move import move_humans, move_robots, update_targets, robot_paths

np.random.seed(421)

num_robots, num_humans, num_goals = np.random.randint([7,   7, -1],
                                                      [10, 10, 0])
env = make_env('simple_herding', benchmark=False,
               num_robots=num_robots, num_humans=num_humans, num_goals=num_goals)
print(num_robots, num_humans, num_goals)

obs_n = env.reset()
act_n = np.zeros([num_humans+num_robots, 2])

# targ_r = np.zeros([num_robots, 2])
# done_r = np.full(num_robots, True) 

# targ_h = np.zeros([num_humans, 2])
# done_h = np.full(num_humans, True) 

# human moves as per spline when outside any herd distance, otherwise only repulsion
# recompute spline when outside herd radius?

traj_n = robot_paths(env.world)

for _ in range(1000):
    # action space is [x,y] where 0 <= x,y <= 1
    targ_h = update_targets(targ_h, done_h)
    ctrl_h, done_h = move_humans(env.world, targ_h)
    act_n[:num_humans] = ctrl_h

    targ_r = update_targets(targ_r, done_r)
    ctrl_r, done_r = move_robots(env.world, targ_r)
    act_n[num_humans:] = ctrl_r

    obs_n, reward_n, done_n, _ = env.step(act_n)

    if np.all(done_n):
        obs_n = env.reset()
        
    env.render()

# implement spline movement
# implement daata collection in different class so it can be used for both files
# dual modality

env.close()