from matplotlib import pyplot as plt
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

def agent_path(agent, keypoints, length=1000, display=False):
    keypoints[:, 0] = agent.state.p_pos

    tck, _ = splprep(keypoints, k=2, s=0) # k=2 makes a c2 spline, s=0 looks good

    t = np.linspace(0, 1, length)
    spline = splev(t, tck)

    if display:
        plt.plot(*keypoints, 'o', label='Original Points')
        plt.plot(*spline, label='Cubic Spline')
        plt.show()

    return spline

def robot_paths(world, bounds=(-1,1), length=1000, num_keypoints=10, disp_idx=None):
    num_robots = len(world.robots)
    # boolean array for which ONE agent path to display
    disp = np.full(num_robots, False) if disp_idx else (np.arange(num_robots) == disp_idx)

    # generate keypoints and paths for robots
    keypoints = np.random.uniform(*bounds, size=[num_robots, 2, num_keypoints])
    return np.array([agent_path(a, k, length=length, display=d) for a,k,d in zip(world.robots, keypoints, disp)])