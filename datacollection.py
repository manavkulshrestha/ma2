import numpy as np
from pathlib import Path

from utility import save_pkl, time_label


def timestep_data(world, robot_actions):
    robots_state = np.array([[*x.state.p_pos, *x.state.p_vel] for x in world.robots])
    humans_state = np.array([[*x.state.p_pos, *x.state.p_vel] for x in world.humans])

    robots_travelled = np.array([x.state.travelled for x in world.robots])
    humans_travelled = np.array([x.state.travelled for x in world.humans])

    return {
        'h_state': humans_state, 'r_state': robots_state,
        'h_trav': humans_travelled, 'r_trav': robots_travelled,
        'r_actions': robot_actions
    }

def save_data(data_arr, num_humans, num_robots, num_goals, *, sub_dir):
    data = {
        'timeseries': data_arr,
        'num_humans': num_humans,
        'num_robots': num_robots,
        'num_goals': num_goals
    }

    save_pkl(data, sub_dir/time_label())

# robots_data = [{'pos':r.pos, 'vel':r.vel, 'action':a, 'travelled': h.travelled} for r,a in zip(robots, actions)]
# humans_data = [{'pos':h.pos, 'vel':h.vel, 'travelled': h.travelled} for h in humans]
    