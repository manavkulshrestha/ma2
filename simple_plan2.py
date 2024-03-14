import numpy as np
from torch_geometric.seed import seed_everything
from torch_geometric.data import Data
import torch
from collections import deque
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

from sim.make_env import make_env
from data.datasets import all_paths, fully_connected, temporal_graph
from sim.utility import load_pkl, pdisp
from nn.networks import LearnedSimulator


seed_everything(740)
WINDOW_LEN = 2
PLAN_RECORDED = False
model_path = 'models/24-03-13-14274015-6635/best_31_0.00044431310379877687.pth'
metadata_path = 'data/spline_i-6635/processed/pop_stats-0_800_800_1000-2.pkl'

reuplsive_range = 0.3


def normalize(x, mu, sigma):
    return (x-mu)/sigma

def denormalize(x, o_mu, o_sigma):
    return x*o_sigma + o_mu

def agent_states(world):
    humans_pos = np.array([h.state.p_pos for h in world.humans])
    humans_vel = np.array([h.state.p_vel for h in world.humans])

    robots_pos = np.array([r.state.p_pos for r in world.robots])
    robots_vel = np.array([r.state.p_vel for r in world.robots])

    return humans_pos, humans_vel, robots_pos, robots_vel

def subgoals(humans_pos, goals_pos, factor=0.1):
    ''' yields unit displacement in the direction of the closest goal for all humans'''
    goals_size = 0.1

    disp_vecs = goals_pos - humans_pos[:, np.newaxis, :]
    distances = np.linalg.norm(disp_vecs, axis=-1)
    closest_gidx = np.argmin(distances, axis=1)

    goal_disps = disp_vecs[np.arange(len(humans_pos)), closest_gidx] # weird, can't do [:, closest_idx]
    unit_goal_disps = goal_disps/np.linalg.norm(goal_disps, axis=1)[:, np.newaxis]
    reached = distances[np.arange(len(humans_pos)), closest_gidx] < goals_size

    return factor * unit_goal_disps, reached

# TODO directly use the one from datasets instead of this
def scene_graph(humans_pos, humans_vel, robots_pos, robots_vel, *, constants):
    robot_mask, edge_index, vel_mu, vel_sigma = constants

    pos = np.vstack([humans_pos, robots_pos])
    vel = np.vstack([humans_vel, robots_vel])

    vel = normalize(vel, vel_mu, vel_sigma)

    disp = torch.tensor(pdisp(pos))
    dist = torch.norm(disp, dim=-1, keepdim=True)

    feats = torch.cat((disp, dist), dim=-1)[tuple(edge_index)]

    graph = Data(
        x = robot_mask.long(), # would need to change when objects, torch.nn.embedding
        # y = torch.tensor(act).float(),
        edge_index = edge_index.long(),
        edge_attr = feats.float(),
        node_dist = feats[:,-1].float(),
        pos = torch.tensor(vel).float(),
        robot_mask=robot_mask.bool()
    )

    return graph

def scene_window(prev_graphs, human_pos, humans_disp, reached, *, constants):
    assert len(prev_graphs) == WINDOW_LEN-1
    robot_mask, edge_index, vel_mu, vel_sigma = constants

    num_robots = torch.count_nonzero(robot_mask)

    nhumans_pos = human_pos + humans_disp
    nhumans_vel = humans_disp
    nrobots_pos = np.zeros([num_robots, 2])
    nrobots_vel = np.zeros([num_robots, 2])

    fg = scene_graph(nhumans_pos, nhumans_vel, nrobots_pos, nrobots_vel, constants=constants)

    return temporal_graph([*prev_graphs, fg], include_y=False, zero_future_states=False) # CURR_IDX = -2/0

    # next_graph = scene_graph(nhumans_pos, nhumans_vel, nrobots_pos, nrobots_vel, constants=constants)
    # next_graph.pos[next_graph.robot_mask] = 0
    # edge_robot_mask = np.full([len(next_graph.x)]*2, False)
    # edge_robot_mask[next_graph.robot_mask] = True
    # edge_robot_mask[:, next_graph.robot_mask] = True
    # edge_robot_mask = edge_robot_mask[tuple(next_graph.edge_index)]
    # next_graph.edge_attr[edge_robot_mask] = 0
    
    # graph_list = [*prev_graphs, next_graph]
    
    # node_feats = torch.cat([g.pos for g in graph_list], dim=-1)
    # edge_feats = torch.cat([g.edge_attr for g in graph_list], dim=-1)
    # node_dists = graph_list[CURR_IDX].node_dist.reshape(-1,1)

    # list_graph = Data(
    #     x = graph_list[0].x,
    #     # y = torch.cat([g.y for g in graph_list], dim=-1),
    #     edge_index = graph_list[0].edge_index,
    #     edge_attr = edge_feats,
    #     node_dist = node_dists,
    #     pos = node_feats,
    #     robot_mask=graph_list[0].robot_mask
    # )

    # return list_graph

def initial_graph(humans_pos, robots_pos, *, constants):
    humans_vel = np.zeros([len(humans_pos), 2])
    robots_vel = np.zeros([len(robots_pos), 2])

    # TODO humans move in the first window_len-1 episodes, add that in?
    return scene_graph(humans_pos, humans_vel, robots_pos, robots_vel, constants=constants)


def set_agent_states(world, timestep, dont=False):
    if not dont:
        for h, state in zip(world.humans, timestep['h_state']):
            h.state.p_pos = state[:2]
            h.state.p_vel = state[2:]

        for r, state in zip(world.robots, timestep['r_state']):
            r.state.p_pos = state[:2]
            r.state.p_vel = state[2:]

    return timestep['r_actions']


def main():
    scene_max = 100000
    human_rng, robot_rng, goal_rng = (4, 5), (4, 5), (1, 2)
    render = True

    # set up environment
    num_humans, num_robots, num_goals = np.random.randint(*np.vstack([human_rng, robot_rng, goal_rng]).T)
    
    # LOAD RECORDED EPISODE
    if PLAN_RECORDED:
        paths = all_paths(6635)
        data = load_pkl(paths[900])
        num_humans, num_robots = [data[x] for x in ['num_humans', 'num_robots']]
        timeseries = data['timeseries']
        num_goals = 1

        act_mse = []
        h_pos_mse = []
        r_pos_mse = []
    # DONE LOADING

    env = make_env('simple_herding', benchmark=False,
                num_humans=num_humans, num_robots=num_robots, num_goals=num_goals)
    num_agents = num_humans+num_robots

    # LOAD RECORDED EPISODE
    if PLAN_RECORDED:
        curr_t = WINDOW_LEN-2
        scene = timeseries[curr_t]
        y_ctrl = set_agent_states(env.world, scene)
    # DONE LOADING
        
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
    env.world.landmarks[0].state.p_pos = [0.8, 0.8]

    for r, pos in zip(env.world.robots, r_poses):
        r.state.p_pos = pos
    for h, pos in zip(env.world.humans, h_poses):
        h.state.p_pos = pos
    
    # variables for environment 
    # obs_n = env.reset()
    done_n = np.full(num_agents, False)
    act_n = np.zeros([num_agents, 2])

    # static info for scene graph
    edge_index = fully_connected(num_agents)
    robot_mask = torch.tensor(np.arange(num_agents) >= num_humans)
    goals_pos = np.array([g.state.p_pos for g in env.world.landmarks])

    # load model
    model = LearnedSimulator(window_size=WINDOW_LEN).cuda()
    # model.load_state_dict(torch.load('models/24-03-02-01474901-6635/best_579_3.0690658604726195e-05.pth')['model'])
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()

    # data population stats from training set
    # metadata = load_pkl('data/spline_i-6635/processed/pop_stats-0_800_800_1000-7.pkl')
    metadata = load_pkl(metadata_path)
    act_mu, act_sigma = metadata['train_act_ms']
    vel_mu, vel_sigma = metadata['train_vel_ms']

    # memory for prior scenes
    humans_pos, _, robots_pos, _ = agent_states(env.world)
    constants = robot_mask, edge_index, vel_mu, vel_sigma
    prev_graphs = deque([initial_graph(humans_pos, robots_pos, constants=constants) for _ in range(WINDOW_LEN-1)])

    # LOAD RECORDED EPISODE
    if PLAN_RECORDED:
        # None in front since it gets popped out
        prev_graphs = deque([None]+[scene_graph(*np.hsplit(s['h_state'], 2), *np.hsplit(s['r_state'], 2), constants=constants) for s in timeseries[:curr_t]])
    # DONE LOADING

    for i in range(scene_max):
        # generate graph and normalized vels, acts. zeros if no data
        states = agent_states(env.world)
        humans_pos = states[0]

        graph = scene_graph(*states, constants=constants)

        # LOAD RECORDED EPISODE
        if PLAN_RECORDED:
            scene = timeseries[curr_t]
            ctrl_y = set_agent_states(env.world, timeseries[curr_t], dont=True)
            graph = scene_graph(*np.hsplit(scene['h_state'], 2), *np.hsplit(scene['r_state'], 2), constants=constants)
        # DONE LOADING

        prev_graphs.popleft()
        prev_graphs.append(graph)

        window = scene_window(prev_graphs, humans_pos, *subgoals(humans_pos, goals_pos), constants=constants)

        # subgoal_test
        # for i, (h, sg, done) in enumerate(zip(env.world.humans, *subgoals(humans_pos, goals_pos))):
        #     if not done:
        #         h.state.p_vel = sg
        #         print(f'setting {i}')

        # LOAD RECORDED EPISODE
        if PLAN_RECORDED:
            if curr_t+1 < len(timeseries):
                fs = timeseries[curr_t+1]
            else:
                break
            fg = scene_graph(*np.hsplit(fs['h_state'], 2), *np.hsplit(fs['r_state'], 2), constants=constants)
            window = temporal_graph([*prev_graphs, fg], include_y=False)
        # DONE LOADING

        out = model(window.cuda())
        ctrl_r = torch.clip(denormalize(out[robot_mask], act_mu, act_sigma), -1, 1).detach().cpu().numpy()
        act_n[num_humans:] = ctrl_r

        # LOAD RECORDED EPISODE
        if PLAN_RECORDED:
            hpos = np.array([h.state.p_pos for h in env.world.humans])
            rpos = np.array([r.state.p_pos for r in env.world.robots])

            act_mse.append(mae(ctrl_y, ctrl_r))
            h_pos_mse.append(mae(scene["h_state"][:,:2], hpos))
            r_pos_mse.append(mae(scene["r_state"][:,:2], rpos))

            print(f'action mae: {act_mse[-1]}')
            print(f'h_posi mae: {h_pos_mse[-1]}')
            print(f'r_posi mae: {r_pos_mse[-1]}')

            print()

            curr_t += 1
        # DONE LOADING
            
        env.step(act_n)

        if np.all(done_n):
            obs_n = env.reset()
            
        if render:
            env.render()

    env.close()

    print('min, mean, max')
    print(f'action mae: {np.min(act_mse)}')
    print(f'h_posi mae: {np.min(h_pos_mse)}')
    print(f'r_posi mae: {np.min(r_pos_mse)}')
    print()

    print(f'action mae: {np.mean(act_mse)}')
    print(f'h_posi mae: {np.mean(h_pos_mse)}')
    print(f'r_posi mae: {np.mean(r_pos_mse)}')
    print()

    print(f'action mae: {np.max(act_mse)}')
    print(f'h_posi mae: {np.max(h_pos_mse)}')
    print(f'r_posi mae: {np.max(r_pos_mse)}')


if __name__ == '__main__':
    main()