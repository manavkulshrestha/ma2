import numpy as np
import time

from sim.make_env import make_env

from sim.move import move_humans, move_robots, robot_paths, update_targets
from data.datacollection import timestep_data


# np.random.seed(421)

# def linear_scenario(scene_len=1000, render=True,
#                     human_rng=(7,10), robot_rng=(7,10), goal_rng=(0,1)):
#     num_humans, num_robots, num_goals = np.random.randint(*np.vstack([human_rng, robot_rng, goal_rng]).T)
#     env = make_env('simple_herding', benchmark=False,
#                 num_humans=num_humans, num_robots=num_robots, num_goals=num_goals)
#     # print(num_robots, num_humans, num_goals)

#     obs_n = env.reset()
#     # robot actcions
#     act_n = np.zeros([num_humans+num_robots, 2])

#     # targets that humans go to and whether they got there 
#     targ_h = np.zeros([num_humans, 2])
#     done_h = np.full(num_humans, True)

#     # targets that robots go to and whether they got there
#     targ_r = np.zeros([num_robots, 2])
#     done_r = np.full(num_robots, True) 

#     timeseries_data = []

#     for _ in range(scene_len):
#         # action space is [x,y] where 0 <= x,y <= 1
#         # if human has reached target, update it, move towards target
#         targ_h = update_targets(targ_h, done_h)
#         ctrl_h, done_h = move_humans(env.world, targ_h)
#         act_n[:num_humans] = ctrl_h

#         # if robot has reached target, update it, move towards target
#         targ_r = update_targets(targ_r, done_r)
#         ctrl_r, done_r = move_robots(env.world, targ_r)
#         act_n[num_humans:] = ctrl_r

#         # apply actions
#         obs_n, reward_n, done_n, _ = env.step(act_n)

#         # done_n not implemenented yet, end condition
#         # if np.all(done_n):
#         #     obs_n = env.reset()
            
#         if render:
#             env.render()

#         timeseries_data.append(timestep_data(env.world, ctrl_h, ctrl_r))

#     env.close()
#     return timeseries_data, num_humans, num_robots, num_goals

def spline_scenario(scene_len=1000, render=True,
                    human_rng=(7,10), robot_rng=(7,10), goal_rng=(0,1),
                    display_spline_idx=None, verbose=False,
                    spline_degree=1, action_noise=0):
    num_humans, num_robots, num_goals = np.random.randint(*np.vstack([human_rng, robot_rng, goal_rng]).T)
    env = make_env('simple_herding', benchmark=False,
                num_humans=num_humans, num_robots=num_robots, num_goals=num_goals, action_noise=action_noise)
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
    path_r = robot_paths(env.world, length=scene_len, disp_idx=display_spline_idx, degree=spline_degree)

    timeseries_data = []

    for i in range(scene_len):
        # action space is [x,y] where 0 <= x,y <= 1
        # if human has reached target, update it, move towards target
        # targ_h = update_targets(targ_h, done_h)
        # ctrl_h, done_h = move_humans(env.world, targ_h)
        # act_n[:num_humans] = ctrl_h

        # if robot has reached target, update it, move towards target
        if verbose:
            print(i)
            
        ctrl_r, _ = move_robots(env.world, path_r[:,:,i])
        act_n[num_humans:] = ctrl_r

        # apply actions
        # print(ctrl_r)
        # print(ctrl_r.min(), ctrl_r.max())
        obs_n, reward_n, done_n, _ = env.step(act_n)

        # done_n not implemenented yet, end condition
        # if np.all(done_n):
        #     obs_n = env.reset()
            
        if render:
            env.render()

        timeseries_data.append(timestep_data(env.world, humans_actions=act_n[:num_humans], robots_actions=ctrl_r))

    env.close()
    return timeseries_data, num_humans, num_robots, num_goals


def main():
    spline_scenario(scene_len=1000, render=True)


if __name__ == '__main__':
    main()