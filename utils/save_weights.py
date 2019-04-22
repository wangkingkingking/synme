import torch
import os.path
def save_weights(model, pkl_file):
    _, ext = os.path.splitext(pkl_file)
    if ext == '.pkl' or '.pth':
        print('Saving state_dict to {:s} ...'.format(pkl_file))
        if isinstance(model, torch.nn.parallel.DataParallel):
            torch.save(model.module.state_dict(), pkl_file)
        else:
            torch.save(model.state_dict(), pkl_file)
    else:
        print('Only support .pkl or .pth now')
