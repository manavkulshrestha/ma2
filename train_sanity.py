from pathlib import Path
import torch
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, NLLLoss, MSELoss, L1Loss
import numpy as np
from torch_geometric import seed_everything
from torch.utils.tensorboard import SummaryWriter

from data.datasets import series_dloaders
from nn.networks import ANet, LearnedSimulator
from sim.utility import time_label
from data.datasets import CURR_IDX

from datetime import datetime


MODEL_PATH = Path('models')
seed = 42 or np.random.randint(1,10000)
seed_everything(seed)
print(f'training with {seed=}')

DEBUG = False
if DEBUG:
    print('DEBUG MODE ON, NOT SAVING ANY CHECKPOINTS OR LOGS')


def save_all(model, optimizer, scheduler, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)

def denormalize(x, o_mu, o_sigma):
    return x*o_sigma + o_mu

def extract_targets(y):
    # assert CURR_IDX < 0
    # return y[:, 2*CURR_IDX:CURR_IDX] <--- incorrect
    assert CURR_IDX > -1
    return y[:, 2*CURR_IDX:2*CURR_IDX+2]

def train_epoch(model, dloader, metadata, loss_fn, opt, *, sch, epoch, progress=True, of_batch=None):
    model.train()
    train_loss = 0
    total_examples = 0

    act_ms = metadata['train_act_ms']

    if of_batch:
        print('OVERFITTING TEST ON ONE BATCH!')
        assert dloader is None
        dloader = [of_batch]

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] training', total=len(dloader)):
        batch = batch.cuda()
        y, mask = batch.y, batch.robot_mask

        opt.zero_grad()

        out = model(batch)
        pred = denormalize(out[mask], *act_ms)
        targ = denormalize(extract_targets(y), *act_ms)

        # get loss and update model
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 
        batch_loss = loss_fn(pred, targ)

        batch_loss.backward()
        opt.step()
        sch.step()

        train_loss += batch_loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return train_loss/total_examples

@torch.no_grad()
def test_epoch(model, dloader, metadata, metric, *, epoch, progress=False):
    model.eval()
    scores = []

    act_ms = metadata['test_act_ms']

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] testing', total=len(dloader)):
        batch = batch.cuda()
        y, mask = batch.y, batch.robot_mask

        out = model(batch)
        pred = denormalize(out[mask].cpu().numpy(), *act_ms)
        targ = denormalize(extract_targets(y).cpu().numpy(), *act_ms)
        
        score = metric(pred, targ)
        scores.append(score)

    return np.mean(scores)

def main():
    epochs = 1000
    batch_sizes = 512, 512
    learning_rate = 1e-4
    tss_rate, window_len = 1, 1

    torch.cuda.empty_cache()

    # assert False
    # data_seed = 6635 #9613
    data_seed = 6279 #5953 #6635 #652 #6635 #652 #9613
    data_folder = f'spline_i-{data_seed}'

    run_path = MODEL_PATH/f'{tl}-{data_seed}'
    run_path.mkdir()

    (train_loader, test_loader), metadata = series_dloaders(
        data_folder,
        chunks=((0, 800), (800, 1000)),
        batch_sizes=batch_sizes,
        shuffles=(True, True),
        timeseries_samplerate=tss_rate,
        window_len=window_len
    )
    of_batch = None # next(iter(train_loader))
    print(metadata)

    # model = ANet(1+2*window_len, 3*window_len, heads=32, concat=False).cuda()
    model = LearnedSimulator(window_size=window_len).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    best_score, best_epoch = np.inf, 0

    for epoch in range(1, epochs):
        train_loss = train_epoch(model, train_loader, metadata=metadata,
                                 loss_fn=MSELoss(), opt=optimizer, sch=scheduler,
                                 of_batch=of_batch, epoch=epoch)
        test_metric = test_epoch(model, test_loader, metadata=metadata,
                                 metric=mse,
                                 epoch=epoch)

        if not DEBUG:
            writer.add_scalar("MSE Loss/train", train_loss, epoch)
            writer.add_scalar("MSE Loss/test", test_metric, epoch)

        if best_score > test_metric:
            best_score = test_metric
            best_epoch = epoch
            if not DEBUG:
                save_all(model, optimizer, scheduler, run_path/f'best_{epoch}_{test_metric}.pth')

        if not DEBUG and epoch % 10 == 0:
            save_all(model, optimizer, scheduler, run_path/f'model_{epoch}.pth')

        print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.10f}, Test MSE: {test_metric:.10f}, best={best_score:.10f} at {best_epoch}')

        if not DEBUG:
            writer.flush()


if __name__ == '__main__':
    tl = time_label()
    with SummaryWriter(f'tblogs/{tl}/') as writer:
        main() 