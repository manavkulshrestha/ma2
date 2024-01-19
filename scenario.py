import numpy as np
import time

from make_env import *

from move import move_humans, move_robots, robot_paths, update_targets
from datacollection import timestep_data


# np.random.seed(421)

def linear_scenario(scene_len=1000, render=True):
    num_humans, num_robots, num_goals = np.random.randint([7,   7, 0],
                                                          [10, 10, 1])
    env = make_env('simple_herding', benchmark=False,
                num_humans=num_humans, num_robots=num_robots, num_goals=num_goals)
    # print(num_robots, num_humans, num_goals)

    obs_n = env.reset()
    # robot actcions
    act_n = np.zeros([num_humans+num_robots, 2])

    # targets that humans go to and whether they got there 
    targ_h = np.zeros([num_humans, 2])
    done_h = np.full(num_humans, True)

    # targets that robots go to and whether they got there
    targ_r = np.zeros([num_robots, 2])
    done_r = np.full(num_robots, True) 

    timeseries_data = []

    for _ in range(scene_len):
        # action space is [x,y] where 0 <= x,y <= 1
        # if human has reached target, update it, move towards target
        targ_h = update_targets(targ_h, done_h)
        ctrl_h, done_h = move_humans(env.world, targ_h)
        act_n[:num_humans] = ctrl_h

        # if robot has reached target, update it, move towards target
        targ_r = update_targets(targ_r, done_r)
        ctrl_r, done_r = move_robots(env.world, targ_r)
        act_n[num_humans:] = ctrl_r

        # apply actions
        obs_n, reward_n, done_n, _ = env.step(act_n)

        # done_n not implemenented yet, end condition
        # if np.all(done_n):
        #     obs_n = env.reset()
            
        if render:
            env.render()

        timeseries_data.append(timestep_data(env.world, act_n))

    env.close()
    return timeseries_data, num_humans, num_robots, num_goals

def spline_scenario(scene_len=1000, render=True):
    num_humans, num_robots, num_goals = np.random.randint([7,   7, 0],
                                                          [10, 10, 1])
    env = make_env('simple_herding', benchmark=False,
                num_humans=num_humans, num_robots=num_robots, num_goals=num_goals)
    # print(num_robots, num_humans, num_goals)

    obs_n = env.reset()
    # robot actcions
    act_n = np.zeros([num_humans+num_robots, 2])

    # targets that humans go to and whether they got there 
    targ_h = np.zeros([num_humans, 2])
    done_h = np.full(num_humans, True)

    # targets that robots go to and whether they got there
    # targ_r = np.zeros([num_robots, 2])
    # done_r = np.full(num_robots, True) 
    path_r = robot_paths(env.world, length=scene_len)

    timeseries_data = []

    for i in range(scene_len):
        # action space is [x,y] where 0 <= x,y <= 1
        # if human has reached target, update it, move towards target
        targ_h = update_targets(targ_h, done_h)
        ctrl_h, done_h = move_humans(env.world, targ_h)
        act_n[:num_humans] = ctrl_h

        # if robot has reached target, update it, move towards target
        ctrl_r, _ = move_robots(env.world, path_r[:,:,i])
        act_n[num_humans:] = ctrl_r

        # apply actions
        obs_n, reward_n, done_n, _ = env.step(act_n)

        # done_n not implemenented yet, end condition
        # if np.all(done_n):
        #     obs_n = env.reset()
            
        if render:
            env.render()

        timeseries_data.append(timestep_data(env.world, act_n))

    env.close()
    return timeseries_data, num_humans, num_robots, num_goals


def main():
    spline_scenario(scene_len=1000, render=True)


if __name__ == '__main__':
    main()