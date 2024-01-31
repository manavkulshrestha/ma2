from pathlib import Path
import torch
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, NLLLoss, MSELoss
import numpy as np

from data.datasets import series_dloaders
from nn.networks import ActionNet
from sim.utility import time_label


MODEL_PATH = Path('models')


def train_epoch(model, dloader, *, opt, epoch, loss_fn, progress=True):
    model.train()
    train_loss = 0
    total_examples = 0

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] training', total=len(dloader)):
        batch = batch.cuda()
        x, edge_idx, y, mask = batch.x, batch.edge_index, batch.y, batch.robot_mask
        opt.zero_grad()

        out = model(x, edge_idx).squeeze()

        # get loss and update model
        batch_loss = loss_fn(out[mask], y)
        batch_loss.backward()
        opt.step()
        train_loss += batch_loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return train_loss/total_examples

@torch.no_grad()
def test_epoch(model, dloader, *, epoch, progress=False):
    model.eval()
    scores = []

    if epoch % 10 == 0:
        pass

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] testing', total=len(dloader)):
        batch = batch.cuda()
        x, edge_idx, y, mask = batch.x, batch.edge_index, batch.y, batch.robot_mask

        out = model(x.float(), edge_idx).squeeze()
        
        score = mse(out[mask].cpu().numpy(), y.cpu().numpy())
        scores.append(score)

    return np.mean(scores)

def main():
    epochs = 10000
    batch_sizes = 16, 16
    learning_rate = 0.001

    torch.cuda.empty_cache()

    run_path = MODEL_PATH/time_label()
    run_path.mkdir()

    train_loader, test_loader = series_dloaders(
        'spline_i-5673',
        chunks=((0, 4000), (4000, 5000)),
        batch_sizes=batch_sizes,
        shuffles=(True, True)
    )

    model = ActionNet(heads=32, concat=False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_score = np.inf

    for epoch in range(1, epochs):
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