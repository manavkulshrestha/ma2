from pathlib import Path
import torch
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, NLLLoss, MSELoss
import numpy as np
from torch_geometric import seed_everything

from data.datasets import series_dloaders
from nn.networks import ANet, LearnedSimulator
from sim.utility import time_label
from data.datasets import CURR_IDX


MODEL_PATH = Path('models')
seed = 42
seed_everything(seed)


def extract_targets(y):
    assert CURR_IDX < 0
    return y[:, 2*CURR_IDX:CURR_IDX]

def train_epoch(model, dloader, *, opt, of_batch, epoch, loss_fn, progress=True):
    model.train()
    train_loss = 0
    total_examples = 0

    batch = of_batch

    progress = tqdm if progress else lambda x, **kwargs: x
    # for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] training', total=len(dloader)):
    batch = batch.cuda()
    y, mask = batch.y, batch.robot_mask

    opt.zero_grad()

    out = model(batch)

    # get loss and update model
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 
    batch_loss = loss_fn(out.reshape(-1, 2)[mask], extract_targets(y))

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
        y, mask = batch.y, batch.robot_mask

        out = model(batch)
        
        score = mse(out.reshape(-1, 2)[mask].cpu().numpy(), extract_targets(y).cpu().numpy())
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

    # model = ANet(1+2*window_len, 3*window_len, heads=32, concat=False).cuda()
    model = LearnedSimulator(window_size=window_len).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    of_batch = next(iter(train_loader))

    best_score = np.inf

    for epoch in range(1, epochs):

        if epoch == 100:
            print()

        train_loss = train_epoch(model, train_loader, opt=optimizer, of_batch=of_batch, epoch=epoch, loss_fn=MSELoss())
        # test_mse = test_epoch(model, test_loader, epoch=epoch)

        # print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.4f}, Test MSE: {test_mse:.4f}')
        print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.4f}')

        # if best_score > test_mse:
        #     best_score = test_mse
        #     torch.save(model.state_dict(), run_path/f'best_{epoch}_{test_mse}.pth')

        if epoch % 10:
            torch.save(model.state_dict(), run_path/f'model_{epoch}.pth')


if __name__ == '__main__':
    main()