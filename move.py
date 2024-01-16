import numpy as np
from scipy.interpolate import splprep, splev


def move_robots(world, targets, kp=1, kd=1, thresh=0.01):
    locs = np.array([x.state.p_pos for x in world.robots])
    errs = targets - locs

    ctrl = kp*errs + kd*(world.robots_eprev - errs)
    done = np.linalg.norm(errs, axis=1) < thresh # ctrl if target vel is 0
    ctrl[done] = 0
    world.robots_eprev = errs

    return ctrl, done

def move_humans(world, targets, kp=1, kd=1, thresh=0.01):
    locs = np.array([x.state.p_pos for x in world.humans])
    errs = targets - locs

    ctrl = kp*errs + kd*(world.humans_eprev - errs)
    done = np.linalg.norm(errs, axis=1) < thresh # ctrl if target vel is 0
    ctrl[done] = 0
    world.humans_eprev = errs

    return ctrl, done

def update_targets(targets, done, bounds=(-1, 1)):
    done_idx, = np.where(done)
    samples = np.random.uniform(*bounds, size=[len(done_idx), 2])
    targets[done_idx] = samples
    return targets

def agent_path(agent, keypoints, bounds=(-1, 1), length=1000):
    keypoints[:, 0] = agent.state.p_pos

    tck, _ = splprep(keypoints, k=2, s=0) # k=2 makes a c2 spline

    t = np.linspace(0, 1, length)
    return splev(t, tck)

def robot_paths(world, bounds=(-1,1), length=1000, num_keypoints=10):
    keypoints = np.random.uniform(*bounds, size=[len(world.robots), 2, num_keypoints])
    return np.array([agent_path(a, k, bounds=bounds, length=length) for a,k in zip(world.robots, keypoints)])