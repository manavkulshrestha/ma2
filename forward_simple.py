import numpy as np
from torch_geometric.seed import seed_everything
from torch_geometric.data import Data
import torch
from collections import deque
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

from sim.make_env import make_env
from data.datasets import all_paths, fully_connected, temporal_graph, get_graph
from sim.utility import load_pkl, pdisp
from nn.networks import ForwardDynamics, LearnedSimulator


seed_everything(42)
WINDOW_LEN = 2
PLAN_RECORDED = False
data_seed = 6296

fmodel_path = 'models/24-04-15-17003528-6296/best_685_3.160200321872253e-06.pth'
metadata_path = f'data/spline_i-{data_seed}/processed/pop_stats-0_800_800_1000-2.pkl'

reuplsive_range = 0.3


def normalize(x, mu, sigma):
    return (x-mu)/sigma

def denormalize(x, o_mu, o_sigma):
    return x*o_sigma + o_mu

def agents_state(world):
    humans_state = np.array([[*h.state.p_pos, *h.state.p_vel] for h in world.humans])
    robots_state = np.array([[*r.state.p_pos, *r.state.p_vel] for r in world.robots])

    return {'h_state': humans_state, 'r_state': robots_state}

def set_agent_states(world, timestep, dont=False):
    if not dont:
        for h, state in zip(world.humans, timestep['h_state']):
            h.state.p_pos = state[:2]
            h.state.p_vel = state[2:]

        for r, state in zip(world.robots, timestep['r_state']):
            r.state.p_pos = state[:2]
            r.state.p_vel = state[2:]

    return timestep['r_actions']

def simple_actions(goal, r_state, num_humans, scale=0.1):
    actions = goal - r_state[:, :2]
    actions = scale * actions/np.linalg.norm(actions, axis=1)[:, np.newaxis]

    return actions

def pad_actions(num_humans, r_actions):
    return np.vstack([np.zeros([num_humans, 2]), r_actions])

def main():
    scene_max = 1000
    human_rng, robot_rng, goal_rng = (4, 5), (4, 5), (1, 2)
    render = True

    # set up environment
    nums = dict(zip(['num_humans', 'num_robots', 'num_goals'], np.random.randint(*np.vstack([human_rng, robot_rng, goal_rng]).T)))
    env = make_env('simple_herding', benchmark=False, **nums, action_noise=0)
    nums['num_agents'] = nums['num_humans']+nums['num_robots']
    del nums['num_goals']
        
    # SET DESIRED SCENARIO
    # rloc = 0.8
    # hloc = rloc - (reuplsive_range-0.1)/np.sqrt(2)
    # r_poses = np.array([(rloc, rloc), (rloc, -rloc), (-rloc, rloc), (-rloc, -rloc)])
    # h_poses = np.array([(hloc, hloc), (hloc, -hloc), (-hloc, hloc), (-hloc, -hloc)])
    # env.world.landmarks[0].state.p_pos = [0, 0]
    rloc = 0.8
    r_poses = np.array([(rloc, rloc), (rloc, -rloc), (-rloc, rloc), (-rloc, -rloc)])
    hd_vecs = np.sign(r_poses)*[-2,-1]
    h_poses = r_poses+hd_vecs/np.linalg.norm(hd_vecs, axis=-1)[:, np.newaxis]*reuplsive_range
    goal_loc = np.array([0, 0.4])
    env.world.landmarks[0].state.p_pos = goal_loc

    for r, pos in zip(env.world.robots, r_poses):
        r.state.p_pos = pos
    for h, pos in zip(env.world.humans, h_poses):
        h.state.p_pos = pos
    
    # variables for environment 
    done_n = np.full(nums['num_agents'], False)
    act_n = np.zeros([nums['num_agents'], 2])

    # load model
    model = ForwardDynamics(window_size=WINDOW_LEN).cuda()
    model.load_state_dict(torch.load(fmodel_path)['model'])
    model.eval()

    # data population stats from training set
    metadata = load_pkl(metadata_path)
    pop_stats = [metadata[f'train_{x}'] for x in ['vel_ms', 'vel_mm', 'act_ms', 'act_mm', 'disp_ms', 'disp_mm', 'dist_ms', 'dist_mm']]

    # memory for prior scenes
    window = []
    for _ in range(WINDOW_LEN):
        state = agents_state(env.world)
        state['r_actions'] = simple_actions(goal_loc, state['r_state'], nums['num_humans'])
        env.step(pad_actions(nums['num_humans'], state['r_actions']))

        #TODO change so humans move sample from env
        window.append(get_graph(state, pop_stats, **nums))
    window = deque(window)

    # some caching/prep
    num_humans = nums['num_humans']
    (disp_mu, disp_sigma), (dist_mu, dist_sigma) = [metadata[f'train_{x}'] for x in ['disp_ms', 'dist_ms']]
    dmu, dsigma = torch.tensor([disp_mu, disp_mu, dist_mu]).cuda(), torch.tensor([disp_sigma, disp_sigma, dist_sigma]).cuda()

    for i in range(scene_max):
        # generate graph and normalized vels, acts. zeros if no data
        state = agents_state(env.world)
        state['r_actions'] = simple_actions(goal_loc, state['r_state'], nums['num_humans'])

        graph = get_graph(state, pop_stats, **nums)

        window.popleft()
        window.append(graph)

        # fix window TODO
        tgraph = temporal_graph(window)

        out = model(tgraph.cuda())
        next_state = torch.clip(denormalize(out, dmu, dsigma), -1, 1).detach().cpu().numpy()

        print(mae(tgraph.future_dis.detach().cpu().numpy(), next_state))
            
        env.step(pad_actions(nums['num_humans'], state['r_actions']))

        if np.all(done_n):
            obs_n = env.reset()
            
        if render:
            env.render()

    env.close()


if __name__ == '__main__':
    main()