import os.path
import torch
def load_weights(model, pkl_file):
    _, ext = os.path.splitext(pkl_file)
    if ext == '.pkl' or '.pth':
        print('Loading {:s} into state_dict ...'.format(pkl_file))
        if isinstance(model, torch.nn.parallel.DataParallel):
            model.module.load_state_dict(torch.load(pkl_file))
        else:
            model.load_state_dict(torch.load(pkl_file))
    else:
        print('Only support .pkl or .pth file')


