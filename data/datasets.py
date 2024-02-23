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

from sim.utility import sliding, chunked, load_pkl, pdisp


CURR_IDX = -2
DATA_PATH = Path('data')


def fully_connected(num_nodes):
    nodes = torch.arange(num_nodes)
    row, col = torch.meshgrid(nodes, nodes, indexing='ij')
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)

    return edge_index


def preprocess(timeseries, skip_val):
    '''
    modifies timeseries dicts to add action_sum,
    downsamples timeseries by skip_val,
    calculates normalization values for vel and act
    '''
    vels, acts = [], []
    timeseries_downsampled = []

    for i, ts in islice(enumerate(timeseries), 10+skip_val, None, skip_val): # skip first 10 frames
        sum_action = np.array([s['r_actions'] for s in timeseries[i-skip_val:i]]).sum(axis=0)

        pos, vel = np.hsplit(ts['r_state'], 2)

        vels.append(vel)
        acts.append(sum_action)

        ts['r_action_sum'] = sum_action
        timeseries_downsampled.append((i, ts))

    vels_stacked = np.vstack(vels)
    acts_stacked = np.vstack(acts)

    return (
        (vels_stacked.mean(), vels_stacked.std()), (vels_stacked.min(), vels_stacked.max()),
        (acts_stacked.mean(), acts_stacked.std()), (acts_stacked.min(), acts_stacked.max()),
        timeseries_downsampled
    )

def get_graph(state, vel_mu, vel_sigma, vel_min, vel_max, act_mu, act_sigma, act_min, act_max, *, num_humans, num_robots, num_agents, normalize=True):
    '''
    normalizes action and velocity
    calculates pairwise displacement vectors
    '''
    states = np.vstack([state['h_state'], state['r_state']])
    (pos, vel), act = np.hsplit(states, 2), state['r_action_sum']

    if normalize: # normalization is agent agnostic using timeseries mean
        # vel = (vel-vel_mu)/vel_sigma
        vel = (vel-vel_min)/(vel_max-vel_min)

        # act = (act-act_mu)/act_sigma
        act = (act-act_min)/(act_max-act_min)

    disp = torch.tensor(pdisp(pos))
    dist = torch.norm(disp, dim=-1, keepdim=True)

    robot_mask = torch.tensor(np.arange(num_agents) >= num_humans) # FIXX TODO TODO
    e_idx = fully_connected(num_agents)
    feats = torch.cat((disp, dist), dim=-1)[tuple(e_idx)]

    # TODO. pred y should be in the middle of the window. Odd window size. predict action to take given result and prior state

    graph = Data(
        x = robot_mask.long(), # would need to change when objects, torch.nn.embedding
        y = torch.tensor(act).float(),
        edge_index = e_idx.long(),
        edge_attr = feats.float(),
        node_dist = feats[:,-1].float(),
        pos = torch.tensor(vel).float(),
        robot_mask=robot_mask.bool()
    )
    
    return graph

def temporal_graph(graph_list):
    # node_feats = torch.cat([graph_list[0].x]+[g.pos for g in graph_list], dim=-1)
    node_feats = torch.cat([g.pos for g in graph_list], dim=-1)
    edge_feats = torch.cat([g.edge_attr for g in graph_list], dim=-1)
    node_dists = graph_list[CURR_IDX].node_dist.reshape(-1,1)

    list_graph = Data(
        x = graph_list[0].x,
        y = torch.cat([g.y for g in graph_list], dim=-1),
        edge_index = graph_list[0].edge_index,
        edge_attr = edge_feats,
        node_dist = node_dists,
        pos = node_feats,
        robot_mask=graph_list[0].robot_mask
    )

    return list_graph


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
        return [f'data_series_dg_{self.root.name}_{self.chunk[0]}-{self.chunk[1]}_{self.tss_rate}.pt']
    
    def process_files(self):
        data_list = []

        start, end = self.chunk
        paths = list(filter(lambda p: p.is_file(), sorted(self.root.iterdir())[start:end]))

        # get population mu, std, min, max (maybe seperated for train and test) only train should be known
        # downsample timeseries

        for path in tqdm(paths, desc=f'Processing'):
            file_data = load_pkl(path)
            nr, nh = file_data['num_robots'], file_data['num_humans']
            na = nr+nh

            vel_ms, vel_mm, act_ms, act_mm, ts = preprocess(file_data['timeseries'], self.tss_rate)
            ts = [get_graph(s, *vel_ms, *vel_mm, *act_ms, *act_mm, num_humans=nh, num_robots=nr, num_agents=na) for i, s in ts]
            for window in chunked(ts, self.window_len):
                data_list.append(temporal_graph(window)) # window is list of graphs, function to combine graphs

        return data_list

    # def process_files(self):
    #     data_list = []
        
    #     start, end = self.chunk
    #     paths = sorted(self.root.iterdir())[start:end]
    #     for path in tqdm(paths, desc=f'Processing'):
    #         file_data = load_pkl(path)

    #         ts = file_data['timeseries']
    #         for i, (prev, curr) in enumerate(sliding(ts[::self.tss_rate], 2)):
    #             # get nodes for prev robots and humans and curr humans
    #             nodes, adj_mat, robot_mask = get_pairgraph_dg(_get_nodes(prev), _get_nodes(curr, include_robots=False))
                
    #             # create coo representation for edges
    #             adj_mat = coo_matrix(adj_mat)
    #             edges = torch.tensor(np.vstack([adj_mat.row, adj_mat.col]))
                
    #             # get actions for robots. N_r x 2 for torque or something in x,y
    #             actions = _get_robots_actions(ts[i:i+self.tss_rate])
                
    #             data = Data(x=nodes.float(), edge_index=edges.long(), y=actions.float(), robot_mask=robot_mask.bool())
    #             data_list.append(data)

    #     return data_list

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
                    hook={}):
    # TODO check with normalized features
    # transform = T.Compose(
    #     # T.NormalizeFeatures(),
    #     # T.ToDevice('cuda')
    # )
    transform = None

    split = zip(['train', 'test'], chunks)
    datasets = [SeriesDataset(DATA_PATH/name, name=n, chunk=c, hook=hook, transform=transform, window_len=window_len, tss_rate=timeseries_samplerate) for n, c in split]
    loaders = [DataLoader(ds, batch_size=bs, shuffle=s, num_workers=8) for ds, bs, s in zip(datasets, batch_sizes, shuffles)]

    return loaders