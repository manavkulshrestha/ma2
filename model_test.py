import numpy as np
import torch
from sklearn.metrics import mean_squared_error as mse

from data.datasets import all_paths, series_dloaders, CURR_IDX
from nn.networks import LearnedSimulator


def denormalize(x, o_mu, o_sigma):
    return x*o_sigma + o_mu

def extract_targets(y):
    assert CURR_IDX < 0
    return y[:, 2*CURR_IDX:CURR_IDX]


def main():
    # paths = all_paths(6635)
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

    pass


if __name__ == '__main__':
    main()