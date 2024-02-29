import numpy as np
from torch_geometric.seed import seed_everything
from torch_geometric.data import Data
import torch
from collections import deque

from sim.make_env import make_env
from data.datasets import fully_connected
from sim.utility import pdisp


seed_everything(42)
WINDOW_LEN = 7


def agent_states(world):
    humans_pos = np.array([h.state.p_pos for h in world.humans])
    robots_pos = np.array([r.state.p_pos for r in world.robots])
    humans_vel = np.array([h.state.p_vel for h in world.humans])
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

def scene_graph(humans_pos, humans_vel, robots_pos, robots_vel, robot_mask, edge_index):
    pos = np.vstack([humans_pos, robots_pos])
    vel = np.vstack([humans_vel, robots_vel])

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

def scene_window(prev_graphs, human_pos, human_disps, reached, robot_mask, edge_index):
    assert len(prev_graphs) == WINDOW_LEN-1
    CURR_IDX = -1

    next_graph = scene_graph()
    graph_list = [*prev_graphs, next_graph]
    
    node_feats = torch.cat([g.pos for g in graph_list], dim=-1)
    edge_feats = torch.cat([g.edge_attr for g in graph_list], dim=-1)
    node_dists = graph_list[CURR_IDX].node_dist.reshape(-1,1)

    list_graph = Data(
        x = graph_list[0].x,
        # y = torch.cat([g.y for g in graph_list], dim=-1),
        edge_index = graph_list[0].edge_index,
        edge_attr = edge_feats,
        node_dist = node_dists,
        pos = node_feats,
        robot_mask=graph_list[0].robot_mask
    )

    return list_graph

def main():
    scene_max = 1000
    human_rng, robot_rng, goal_rng = (7, 10), (7, 10), (1, 5)
    render = True

    # set up environment
    num_humans, num_robots, num_goals = np.random.randint(*np.vstack([human_rng, robot_rng, goal_rng]).T)
    env = make_env('simple_herding', benchmark=False,
                num_humans=num_humans, num_robots=num_robots, num_goals=num_goals)
    num_agents = num_humans+num_robots
    
    # variables for environment 
    obs_n = env.reset()
    done_n = np.full(num_agents, False)
    act_n = np.zeros_like(done_n)

    # static info for scene graph
    edge_index = fully_connected(num_agents)
    robot_mask = torch.tensor(np.arange(num_agents) >= num_humans)
    goals_pos = np.array([g.state.p_pos for g in env.world.landmarks])

    # load model
    # TODO

    # memory for prior positions
    prev_graphs = deque([])

    for i in range(scene_max):
        # generate graph and normalized vels, acts. zeros if no data
        states = agent_states(env.world)

        graph = scene_graph(*states, robot_mask, edge_index)
        prev_graphs.popleft()
        prev_graphs.append(graph)

        window = scene_window(prev_graphs, *subgoals(states[0], goals_pos), robot_mask, edge_index)

        ctrl_r = model(window)
        act_n[num_humans:] = ctrl_r

        # subgoal_test
        # for i, (h, sg, done) in enumerate(zip(env.world.humans, *subgoals(env.world))):
        #     if not done:
        #         h.state.p_vel = sg
        #         print(f'setting {i}')

        env.step(act_n)

        if np.all(done_n):
            obs_n = env.reset()
            
        if render:
            env.render()

    env.close()


if __name__ == '__main__':
    main()