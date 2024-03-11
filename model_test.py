import numpy as np
import torch
from sklearn.metrics import mean_squared_error as mse

from data.datasets import all_paths, downsample, get_graph, series_dloaders, CURR_IDX, temporal_graph
from nn.networks import LearnedSimulator
from sim.utility import chunked, load_pkl


def denormalize(x, o_mu, o_sigma):
    return x*o_sigma + o_mu

def extract_targets(y):
    assert CURR_IDX < 0
    return y[:, 2*CURR_IDX:CURR_IDX]


def dataset_test():
    data_seed = 6635
    data_folder = f'spline_i-{data_seed}'
    batch_sizes = 1,1
    tss_rate = 1
    window_len = 7

    (train_loader, test_loader), metadata = series_dloaders(
        data_folder,
        chunks=((0, 800), (800, 1000)),
        batch_sizes=batch_sizes,
        shuffles=(True, True),
        timeseries_samplerate=tss_rate,
        window_len=window_len
    )
    act_ms = metadata['train_act_ms']

    model = LearnedSimulator(window_size=window_len).cuda()
    model.load_state_dict(torch.load('models/24-03-02-01474901-6635/best_579_3.0690658604726195e-05.pth')['model'])
    model.eval()

    for batch in test_loader:
        batch = batch.cuda()
        y, mask = batch.y, batch.robot_mask

        out = model(batch)
        pred = denormalize(out[mask].detach().cpu().numpy(), *act_ms)
        targ = denormalize(extract_targets(y).cpu().numpy(), *act_ms)
        
        score = mse(pred, targ)
        print(score)

        print()

def file_test():
    data_seed = 6635
    data_folder = f'spline_i-{data_seed}'
    paths = all_paths(data_seed)
    window_len = 7
    batch_sizes = 1,1
    tss_rate = 1
    _, metadata = series_dloaders(
        data_folder,
        chunks=((0, 800), (800, 1000)),
        batch_sizes=batch_sizes,
        shuffles=(True, True),
        timeseries_samplerate=tss_rate,
        window_len=window_len
    )
    data = load_pkl(paths[900])
    nh, nr = [data[x] for x in ['num_humans', 'num_robots']]
    na = nh+nr

    model = LearnedSimulator(window_size=window_len).cuda()
    model.load_state_dict(torch.load('models/24-03-02-01474901-6635/best_579_3.0690658604726195e-05.pth')['model'])
    model.eval()
    
    act_ms = metadata['train_act_ms']
    act_mm = metadata['train_act_mm']
    vel_ms = metadata['train_vel_ms']
    vel_mm = metadata['train_vel_mm']

    data_list = []
    ts = downsample(data['timeseries'], tss_rate)
    ts = [get_graph(s, *vel_ms, *vel_mm, *act_ms, *act_mm, num_humans=nh, num_robots=nr, num_agents=na) for i, s in ts]
    for window in chunked(ts, window_len):
        data_list.append(temporal_graph(window)) # window is list of graphs, function to combine graphs

    for batch in data_list:
        batch = batch.cuda()
        y, mask = batch.y, batch.robot_mask

        out = model(batch)
        pred = denormalize(out[mask].detach().cpu().numpy(), *act_ms)
        targ = denormalize(extract_targets(y).cpu().numpy(), *act_ms)
        
        score = mse(pred, targ)
        print(score)

        print()


def main():
    data_seed = 6635
    paths = all_paths(data_seed)
    data = load_pkl(paths[900])
    nh, nr = [data[x] for x in ['num_humans', 'num_robots']]
    na = nh+nr

    

if __name__ == '__main__':
    main()