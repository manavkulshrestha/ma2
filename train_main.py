from pathlib import Path
import torch
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, NLLLoss, MSELoss
import numpy as np

from data.datasets import series_dloaders
from nn.networks import ANet
from sim.utility import time_label


MODEL_PATH = Path('models')
# torch.set_default_dtype(torch.float64)


def train_epoch(model, dloader, *, opt, epoch, loss_fn, progress=True):
    model.train()
    train_loss = 0
    total_examples = 0

    batch = next(iter(dloader))

    progress = tqdm if progress else lambda x, **kwargs: x
    # for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] training', total=len(dloader)):
    batch = batch.cuda()
    x, e_idx, e_atr, y, mask = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.robot_mask

    opt.zero_grad()

    out = model(x, e_idx, e_atr)

    # get loss and update model
    batch_loss = loss_fn(out.reshape(-1, 2)[mask], y[:,-2:])

    batch_loss.backward()
    opt.step()

    train_loss += batch_loss.item() * batch.num_graphs
    total_examples += batch.num_graphs

    return train_loss/total_examples

@torch.no_grad()
def test_epoch(model, dloader, *, epoch, progress=False):
    model.eval()
    scores = []

    # batch = next(iter(dloader))

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] testing', total=len(dloader)):
        batch = batch.cuda()
        x, e_idx, e_atr, y, mask = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.robot_mask

        out = model(x, e_idx, e_atr)
        
        score = mse(out.reshape(-1, 2)[mask].cpu().numpy(), y[:,-2:].cpu().numpy())
        scores.append(score)

        return np.mean(scores)

def main():
    epochs = 1000
    batch_sizes = 1, 1
    learning_rate = 0.001
    tss_rate, window_len = 5, 7

    torch.cuda.empty_cache()

    data_seed = 1691
    data_folder = f'spline_i-{data_seed}'

    run_path = MODEL_PATH/f'{time_label()}-{data_seed}'
    run_path.mkdir()

    train_loader, test_loader = series_dloaders(
        data_folder,
        chunks=((0, 80), (80, 100)),
        batch_sizes=batch_sizes,
        shuffles=(True, True),
        timeseries_samplerate=tss_rate,
        window_len=window_len
    )

    model = ANet(1+2*window_len, 3*window_len, heads=32, concat=False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_score = np.inf

    for epoch in range(1, epochs):

        if epoch == 100:
            print()

        train_loss = train_epoch(model, train_loader, opt=optimizer, epoch=epoch, loss_fn=MSELoss())
        test_mse = test_epoch(model, test_loader, epoch=epoch)

        print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.4f}, Test MSE: {test_mse:.4f}')

        if best_score > test_mse:
            best_score = test_mse
            torch.save(model.state_dict(), run_path/f'best_{epoch}_{test_mse}.pth')

        if epoch % 10:
            torch.save(model.state_dict(), run_path/f'model_{epoch}.pth')


if __name__ == '__main__':
    main()