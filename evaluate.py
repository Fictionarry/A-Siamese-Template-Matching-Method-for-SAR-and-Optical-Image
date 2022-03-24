import torch
import torch.nn.functional as F
from tqdm import tqdm


def loss_fn(batch, pred):
    loss = torch.tensor(0).to(device=pred.device, dtype=torch.float32)
    # print(pred.shape)
    for loc, response_map in zip(batch['loc'], pred):

        loc = loc.cpu().numpy()
        sz = response_map.shape[0]
        response_map = torch.sigmoid(response_map)
        pred_area = response_map[int(sz * loc[0]) : int(sz * loc[2]), int(sz * loc[1]) : int(sz * loc[3])]

        alpha = pred_area.shape[0] ** 2 / sz ** 2
        
        loss += -(1 - alpha) * pred_area.sum() + alpha * (response_map.sum() - pred_area.sum())

    return loss / pred.shape[0]



def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    loss = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        
        batch['template'] = batch['template'].to(device=device, dtype=torch.float32)
        batch['search'] = batch['search'].to(device=device, dtype=torch.float32)


        with torch.no_grad():
            pred = net(batch)
            loss += loss_fn(batch, pred)
           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return loss
    return loss / num_val_batches
