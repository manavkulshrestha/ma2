import time
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


CURR_IDX = 0
DATA_PATH = Path('data')


def all_paths(seed, start=None, end=None):
    return list(filter(lambda p: p.is_file() and p.suffix == '.pkl', sorted(Path(f'data/spline_i-{seed}').iterdir())[start:end]))

def fully_connected(num_nodes):
    nodes = torch.arange(num_nodes)
    row, col = torch.meshgrid(nodes, nodes, indexing='ij')
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)

    return edge_index

def downsample(timeseries, skip_val):
    '''
    modifies timeseries dicts to add action_sum,
    downsamples timeseries by skip_val,
    calculates normalization values for vel and act
    '''
    # vels, acts = [], []
    timeseries_downsampled = []

    for i, ts in islice(enumerate(timeseries), 10+skip_val, None, skip_val): # skip first 10 frames
        sum_action = np.array([s['r_actions'] for s in timeseries[i-skip_val:i]]).sum(axis=0)

        # pos, vel = np.hsplit(ts['r_state'], 2)

        # vels.append(vel)
        # acts.append(sum_action)

        ts['r_action_sum'] = sum_action
        timeseries_downsampled.append((i, ts))

    # vels_stacked = np.vstack(vels)
    # acts_stacked = np.vstack(acts)

    # return (
    #     (vels_stacked.mean(), vels_stacked.std()), (vels_stacked.min(), vels_stacked.max()),
    #     (acts_stacked.mean(), acts_stacked.std()), (acts_stacked.min(), acts_stacked.max()),
    #     timeseries_downsampled
    # )

    return timeseries_downsampled

def get_graph(state, vel_mu, vel_sigma, vel_min, vel_max, act_mu, act_sigma, act_min, act_max, *, num_humans, num_robots, num_agents, normalize=True,
              exclude_humans=True):
    '''
    normalizes action and velocity
    calculates pairwise displacement vectors
    '''
    # states = np.vstack([state['h_state'], state['r_state']])
    if exclude_humans:
        states = state['r_state']
    else:
        states = np.vstack([state['h_state'], state['r_state']])

    (pos, vel), act = np.hsplit(states, 2), state['r_action_sum']

    if normalize: # normalization is agent agnostic using timeseries mean
        vel = (vel-vel_mu)/vel_sigma
        act = (act-act_mu)/act_sigma
        # vel = (vel-vel_min)/(vel_max-vel_min)
        # act = (act-act_min)/(act_max-act_min)

    disp = torch.tensor(pdisp(pos))
    dist = torch.norm(disp, dim=-1, keepdim=True)

    if exclude_humans:
        robot_mask = torch.full([num_robots], True)
        e_idx = fully_connected(num_robots)
    else:
        robot_mask = torch.arange(num_agents) >= num_humans
        e_idx = fully_connected(num_agents)
    

    feats = torch.cat((disp, dist), dim=-1)[tuple(e_idx)]

    graph = Data(
        x = robot_mask.long(), # would need to change when objects, torch.nn.embedding
        y = torch.tensor(act).float(),
        edge_index = e_idx.long(),
        edge_attr = feats.float(),
        node_dist = feats[:,-1].float(),
        pos = torch.tensor(vel).float(),
        robot_mask=robot_mask.bool()
    )

    # if zero_humans:
    #     human_mask = graph.robot
    #     # modify pos to [0,0] out all related to humans
    #     graph.pos[human_mask] = 0
    #     # modify edge_attr to [0,0,0] out all related to humans
    #     edge_human_mask = np.full([len(graph.x)]*2, False)
    #     edge_human_mask[human_mask] = True
    #     edge_human_mask[:, human_mask] = True
    #     edge_human_mask = edge_human_mask[tuple(graph.edge_index)]
    #     graph.edge_attr[edge_human_mask] = 0
    
    return graph

def temporal_graph(graph_list, include_y=True, zero_future_states=False, has_future=True):
    # node_feats = torch.cat([graph_list[0].x]+[g.pos for g in graph_list], dim=-1)
    if has_future and len(graph_list) > 1:
        for fg in graph_list[CURR_IDX+1:]:
            # modify pos to [0,0] out all related to robots
            fg.pos[fg.robot_mask] = 0
            # modify edge_attr to [0,0,0] out all related to robots
            edge_robot_mask = np.full([len(fg.x)]*2, False)
            edge_robot_mask[fg.robot_mask] = True
            edge_robot_mask[:, fg.robot_mask] = True
            edge_robot_mask = edge_robot_mask[tuple(fg.edge_index)]
            fg.edge_attr[edge_robot_mask] = 0

            if zero_future_states:
                fg.edge_attr[:,:] = -0
                fg.pos[:,:] = -0

    node_feats = torch.cat([g.pos for g in graph_list], dim=-1)
    edge_feats = torch.cat([g.edge_attr for g in graph_list], dim=-1)
    node_dists = graph_list[CURR_IDX].node_dist.reshape(-1,1)

    data_kwargs = {
        'x': graph_list[0].x,
        # 'y': torch.cat([g.y for g in graph_list], dim=-1),
        'edge_index': graph_list[0].edge_index,
        'edge_attr': edge_feats,
        'node_dist': node_dists,
        'pos': node_feats,
        'robot_mask': graph_list[0].robot_mask
    }
    if include_y:
        data_kwargs['y'] = torch.cat([g.y for g in graph_list], dim=-1)
    list_graph = Data(**data_kwargs)

    return list_graph

def population_stats(file_paths, skip_val):
    # TODO should human vels also be considered
    vels = []
    acts = []

    for path in tqdm(file_paths, desc='Calculating Population Stats'):
        timeseries = load_pkl(path)['timeseries']
        ts = islice(enumerate(timeseries), 20+skip_val, None, skip_val)
        for i, t in ts:
            sum_action = np.array([s['r_actions'] for s in timeseries[i-skip_val:i]]).sum(axis=0)
            _, vel = np.hsplit(t['r_state'], 2)

            vels.append(vel)
            acts.append(sum_action)

    vels_stacked = np.vstack(vels)
    acts_stacked = np.vstack(acts)

    return (
        (vels_stacked.mean(), vels_stacked.std()),
        (vels_stacked.min(), vels_stacked.max()),
        (acts_stacked.mean(), acts_stacked.std()),
        (acts_stacked.min(), acts_stacked.max()),
    )

class SeriesDataset(InMemoryDataset):
    def __init__(self, root, name, *, chunk, tss_rate, window_len, hook={},
                 transform=None, pre_transform=None, pre_filter=None):
        self.chunk = chunk
        self.root = Path(root)
        self.tss_rate = tss_rate
        self.window_len = window_len
        self.hook = hook
        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data_series_dg_{self.root.name}_{self.chunk[0]}-{self.chunk[1]}_{self.tss_rate}_{self.window_len}.pt']
    
    def process_files(self):
        data_list = []

        start, end = self.chunk
        paths = list(filter(lambda p: p.is_file(), sorted(self.root.iterdir())[start:end]))

        # get population mu, std, min, max (maybe seperated for train and test) only train should be known
        vel_ms, vel_mm, act_ms, act_mm = population_stats(paths, self.tss_rate)
        self.hook[f'{self.name}_vel_mm'] = vel_mm
        self.hook[f'{self.name}_act_mm'] = act_mm
        self.hook[f'{self.name}_vel_ms'] = vel_ms
        self.hook[f'{self.name}_act_ms'] = act_ms

        for path in tqdm(paths, desc=f'Processing'):
            file_data = load_pkl(path)
            nr, nh = file_data['num_robots'], file_data['num_humans']
            na = nr+nh

            ts = downsample(file_data['timeseries'], self.tss_rate)
            ts = [get_graph(s, *vel_ms, *vel_mm, *act_ms, *act_mm, num_humans=nh, num_robots=nr, num_agents=na) for i, s in ts]
            for window in chunked(ts, self.window_len):
                data_list.append(temporal_graph(window)) # window is list of graphs, function to combine graphs

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

    datasets = [SeriesDataset(path, name=n, chunk=c, hook=metadata, transform=transform, window_len=window_len, tss_rate=timeseries_samplerate) for n, c in split]
    loaders = [DataLoader(ds, batch_size=bs, shuffle=s, num_workers=8) for ds, bs, s in zip(datasets, batch_sizes, shuffles)]

    hook_path = path/f'processed/pop_stats-{"_".join(map(str, np.array(chunks).flatten()))}-{window_len}.pkl'
    if len(metadata):
        save_pkl(metadata, hook_path)
    else:
        metadata = load_pkl(hook_path)
        assert len(metadata) != 0

    return loaders, metadata