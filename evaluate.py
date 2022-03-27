import torch
import torch.nn.functional as F
from tqdm import tqdm


def loss_fn(batch, pred):
    loss = torch.tensor(0).to(device=pred.device, dtype=torch.float32)
    # print(pred.shape)
    for loc, response_map in zip(batch['loc'], pred):

        loc = loc.cpu().numpy()
        response_map = response_map[0]
        sz_x = response_map.shape[0]
        sz_y = response_map.shape[1]
        # print(sz)

        pred_radius = 10
        response_map = torch.sigmoid(response_map)
        template_scale_x = loc[2] - loc[0]
        template_scale_y = loc[3] - loc[1]
        loc_y = int(sz_y * (loc[1] + loc[3] - template_scale_y) / 2)
        loc_x = int(sz_x * (loc[0] + loc[2] - template_scale_x) / 2)
        pred_area = response_map[max(0, loc_y - pred_radius) : min(sz_y, loc_y + pred_radius),  max(0, loc_x - pred_radius) : min(sz_x, loc_x + pred_radius)]
        alpha = (pred_area.shape[0] * pred_area.shape[1]) / (sz_x * sz_y)
        print(pred_area.shape)
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
