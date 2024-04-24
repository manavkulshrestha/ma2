import copy
import time
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
# from scipy.spatial.distance import pdist
from itertools import islice

from sim.utility import save_pkl, sliding, chunked, load_pkl, pdisp
import matplotlib.patches as patches

CURR_IDX = 0
DATA_PATH = Path('data')


def all_paths(seed, start=None, end=None):
    return list(filter(lambda p: p.is_file() and p.suffix == '.pkl', sorted(Path(f'data/spline_i-{seed}').iterdir())[start:end]))

def fully_connected(num_nodes):
    nodes = torch.arange(num_nodes)
    row, col = torch.meshgrid(nodes, nodes, indexing='ij')
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)

    return edge_index

# def downsample(timeseries, skip_val):
#     '''
#     modifies timeseries dicts to add action_sum,
#     downsamples timeseries by skip_val,
#     calculates normalization values for vel and act
#     '''
#     # vels, acts = [], []
#     timeseries_downsampled = []

#     for i, ts in islice(enumerate(timeseries), 10+skip_val, None, skip_val): # skip first 10 frames
#         sum_action = np.array([s['r_actions'] for s in timeseries[i-skip_val:i]]).sum(axis=0)

#         # pos, vel = np.hsplit(ts['r_state'], 2)

#         # vels.append(vel)
#         # acts.append(sum_action)

#         ts['r_action_sum'] = sum_action
#         timeseries_downsampled.append((i, ts))

#     # vels_stacked = np.vstack(vels)
#     # acts_stacked = np.vstack(acts)

#     # return (
#     #     (vels_stacked.mean(), vels_stacked.std()), (vels_stacked.min(), vels_stacked.max()),
#     #     (acts_stacked.mean(), acts_stacked.std()), (acts_stacked.min(), acts_stacked.max()),
#     #     timeseries_downsampled
#     # )

#     return timeseries_downsampled

def get_graph(state, pop_stats, *, num_humans, num_robots, num_agents, normalize=True,
              no_vel=False):
    '''
    normalizes action and velocity
    calculates pairwise displacement vectors
    '''
    vel_ms, vel_mm, act_ms, act_mm, disp_ms, disp_mm, dist_ms, dist_mm = pop_stats
    vel_mu, vel_sigma = vel_ms
    act_mu, act_sigma = act_ms
    disp_mu, disp_sigma = disp_ms
    dist_mu, dist_sigma = dist_ms

    states = np.vstack([state['h_state'], state['r_state']])

    (pos, vel), act = np.hsplit(states, 2), state['r_actions']
    disp = torch.tensor(pdisp(pos))
    dist = torch.norm(disp, dim=-1, keepdim=True)

    # this could probably be cached and passed directly
    robot_mask = torch.arange(num_agents) >= num_humans
    e_idx = fully_connected(num_agents)

    feats = torch.cat((disp, dist), dim=-1)[tuple(e_idx)]

    vel = torch.tensor(vel).float()
    act = torch.tensor(act).float()

     # normalization is agent agnostic using timeseries mean
    if normalize:
        vel = (vel-vel_mu)/vel_sigma
        act = (act-act_mu)/act_sigma

        feat_mu = torch.tensor([disp_mu, disp_mu, dist_mu])
        feat_sigma = torch.tensor([disp_sigma, disp_sigma, dist_sigma])
        feats = (feats-feat_mu)/feat_sigma

    vels_acts = torch.zeros([len(robot_mask), vel.shape[1]+act.shape[1]])
    vels_acts[:,:2] = vel
    vels_acts[robot_mask, 2:] = act

    graph = Data(
        x = robot_mask.long(),
        y = act.float(),
        edge_index = e_idx.long(),
        edge_attr = feats.float(),
        node_dist = feats[:,-1].float(),
        pos = None if no_vel else vel.float(),
        robot_mask = robot_mask.bool(),
        vels_acts = vels_acts.float(),
        vah_mask = (vels_acts == 0).bool()
    )
    
    return graph

def temporal_graph(graph_list, include_y=True, no_vel=False):
    graph_list = list(graph_list)

    prev_graphs = copy.deepcopy(graph_list[:CURR_IDX+1])
    futu_graphs = copy.deepcopy(graph_list[CURR_IDX+1:]) if CURR_IDX+1 < len(graph_list) else None
    futh_graphs = copy.deepcopy(futu_graphs) # will only contain human info

    for fg in futh_graphs:
        # modify pos to [0,0] out all related to robots
        fg.pos[fg.robot_mask] = 0
        # modify edge_attr to [0,0,0] out all related to robots
        edge_robot_mask = np.full([len(fg.x)]*2, False)
        edge_robot_mask[fg.robot_mask] = True
        edge_robot_mask[:, fg.robot_mask] = True
        edge_robot_mask = edge_robot_mask[tuple(fg.edge_index)]
        fg.edge_attr[edge_robot_mask] = 0
        # TODO add mask for this to graph so that it can be rezerod after noise addition
        # ^Needed for backward model

    node_feats = None if no_vel else torch.cat([g.pos for g in graph_list], dim=-1) # node features for backward model ??
    bedge_feats = torch.cat([g.edge_attr for g in prev_graphs+futh_graphs], dim=-1) # edge features for backward model
    act = torch.cat([g.y for g in graph_list], dim=-1) # ground truth for backward model

    fedge_feats = torch.cat([g.edge_attr for g in prev_graphs], dim=-1) # edge features for forward model
    vels_acts = torch.cat([g.vels_acts for g in prev_graphs], dim=-1) # node features for forward model
    future_dis = torch.cat([g.edge_attr for g in futu_graphs], dim=-1) # ground truth for forward model

    node_dists = graph_list[CURR_IDX].node_dist.reshape(-1,1)

    data_kwargs = {
        'x': graph_list[0].x,
        # 'y': torch.cat([g.y for g in graph_list], dim=-1),
        'edge_index': graph_list[0].edge_index,
        'bedge_attr': bedge_feats,
        'fedge_attr': fedge_feats,
        'future_dis': future_dist,
        'vels_acts': vels_acts,
        'node_dist': node_dists,
        'pos': node_feats,
        'robot_mask': graph_list[0].robot_mask,
        'vah_mask': graph_list[0].vah_mask # which act values to zero (associated with humans)
    }
    if include_y:
        data_kwargs['y'] = act
    list_graph = Data(**data_kwargs)

    return list_graph

def show_frame(hpos, rpos, maxlim=2):
    fig, ax = plt.subplots()
    
    # Plot yellow circles for hpos
    for i in range(len(hpos)):
        ax.add_patch(plt.Circle((hpos[i][0], hpos[i][1]), 0.05, color='yellow', fill=True))
    
    # Plot red circles for rpos
    for j in range(len(rpos)):
        ax.add_patch(plt.Circle((rpos[j][0], rpos[j][1]), 0.05, color='red', fill=True))

    box = patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='black', fill=False)
    ax.add_patch(box)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-maxlim, maxlim)
    ax.set_ylim(-maxlim, maxlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Simulation Frame')
    plt.show()

def population_stats(file_paths):
    # TODO should human vels also be considered
    rvels = []
    acts = []
    disps = []
    dists = []
    hvels = []

    for i, path in enumerate(tqdm(file_paths, desc='Calculating Population Stats')):
        for j, s in enumerate(load_pkl(path)['timeseries']):
            ract = s['r_actions']
            rpos, rvel = np.hsplit(s['r_state'], 2)
            hpos, hvel = np.hsplit(s['h_state'], 2)

            rvels.append(rvel)
            acts.append(ract)
            hvels.append(hvel)

            disp = pdisp(np.vstack([hpos, rpos]))
            dist = np.linalg.norm(disp, axis=-1, keepdims=True)

            # if (disp > 10).any():
            #     np.set_printoptions(suppress=True)
            #     print(i, j, disp) #TODO check why disp is high even when agents in range of each other within box
            #     # TODO implement above fix in data collector
            #     # TODO scale disp as per dimension specific mean and std (may not be necessary after first fix)
            #     show_frame(hpos, rpos)
            # np.set_printoptions(suppress=False)

            disps.append(disp.reshape(-1,2))
            dists.append(dist.flatten())

    rvels_stacked = np.vstack(rvels)
    acts_stacked = np.vstack(acts)
    disps_stacked = np.vstack(disps)
    dists_stacked = np.concatenate(dists)

    return (
        (rvels_stacked.mean(), rvels_stacked.std()),
        (rvels_stacked.min(), rvels_stacked.max()),

        (acts_stacked.mean(), acts_stacked.std()),
        (acts_stacked.min(), acts_stacked.max()),

        (disps_stacked.mean(), disps_stacked.std()),
        (disps_stacked.min(), disps_stacked.max()),

        (dists_stacked.mean(), dists_stacked.std()),
        (dists_stacked.min(), dists_stacked.max()),
    )

class SeriesDataset(InMemoryDataset):
    def __init__(self, root, name, *, chunk, tss_rate, window_len, hook={},
                 transform=None, pre_transform=None, pre_filter=None, no_vel=False,
                 noise_factor=0):
        self.chunk = chunk
        self.root = Path(root)
        self.tss_rate = tss_rate
        self.window_len = window_len
        self.hook = hook
        self.name = name
        self.no_vel = no_vel
        self.noise_factor = noise_factor

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data_series_dg_{self.root.name}_{self.chunk[0]}-{self.chunk[1]}_{self.tss_rate}_{self.window_len}_{"novel" if self.no_vel else ""}.pt']
    
    def process_files(self):
        data_list = []

        start, end = self.chunk
        paths = list(filter(lambda p: p.is_file(), sorted(self.root.iterdir())[start:end]))

        # get population mu, std, min, max (maybe seperated for train and test) only train should be known
        pop_stats = population_stats(paths)
        vel_ms, vel_mm, act_ms, act_mm, disp_ms, disp_mm, dist_ms, dist_mm = pop_stats
        self.hook[f'{self.name}_vel_mm'] = vel_mm
        self.hook[f'{self.name}_act_mm'] = act_mm
        self.hook[f'{self.name}_vel_ms'] = vel_ms
        self.hook[f'{self.name}_act_ms'] = act_ms

        self.hook[f'{self.name}_disp_ms'] = disp_ms
        self.hook[f'{self.name}_disp_mm'] = disp_mm
        self.hook[f'{self.name}_dist_ms'] = dist_ms
        self.hook[f'{self.name}_dist_mm'] = dist_mm

        for path in tqdm(paths, desc=f'Processing'):
            file_data = load_pkl(path)
            nr, nh = file_data['num_robots'], file_data['num_humans']
            na = nr+nh

            ts = file_data['timeseries']
            ts = [get_graph(s, pop_stats, num_humans=nh, num_robots=nr, num_agents=na, no_vel=self.no_vel) for s in ts]
            for window in chunked(ts, self.window_len):
                data_list.append(temporal_graph(window, no_vel=self.no_vel)) # window is list of graphs, function to combine graphs

        return data_list

    def process(self):
        data_list = self.process_files()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def series_dloaders(name,
                    chunks=((0, 80), (80, 100)),
                    batch_sizes=(64, 64),
                    shuffles=(True, True),
                    timeseries_samplerate=1,
                    window_len=7,
                    noise_factor=0):
    # TODO check with normalized features
    # transform = T.Compose(
    #     # T.NormalizeFeatures(),
    #     # T.ToDevice('cuda')
    # )
    transform = None

    metadata = {}
    path = DATA_PATH/name
    split = zip(['train', 'test'], chunks)

    datasets = [SeriesDataset(path, name=n, chunk=c, hook=metadata, transform=transform, window_len=window_len, tss_rate=timeseries_samplerate, noise_factor=noise_factor) for n, c in split]
    loaders = [DataLoader(ds, batch_size=bs, shuffle=s, num_workers=8) for ds, bs, s in zip(datasets, batch_sizes, shuffles)]

    hook_path = path/f'processed/pop_stats-{"_".join(map(str, np.array(chunks).flatten()))}-{window_len}.pkl'
    if len(metadata):
        save_pkl(metadata, hook_path)
    else:
        metadata = load_pkl(hook_path)
        assert len(metadata) != 0

    return loaders, metadata

def posseries_dloaders(name,
                    chunks=((0, 80), (80, 100)),
                    batch_sizes=(64, 64),
                    shuffles=(True, True),
                    timeseries_samplerate=1,
                    window_len=7):
    # TODO check with normalized features
    # transform = T.Compose(
    #     # T.NormalizeFeatures(),
    #     # T.ToDevice('cuda')
    # )
    transform = None

    metadata = {}
    path = DATA_PATH/name
    split = zip(['train', 'test'], chunks)

    datasets = [SeriesDataset(path, name=n, chunk=c, hook=metadata, transform=transform, window_len=window_len, tss_rate=timeseries_samplerate, no_vel=True) for n, c in split]
    loaders = [DataLoader(ds, batch_size=bs, shuffle=s, num_workers=8) for ds, bs, s in zip(datasets, batch_sizes, shuffles)]

    hook_path = path/f'processed/pop_stats-{"_".join(map(str, np.array(chunks).flatten()))}-{window_len}.pkl'
    if len(metadata):
        save_pkl(metadata, hook_path)
    else:
        metadata = load_pkl(hook_path)
        assert len(metadata) != 0

    return loaders, metadata