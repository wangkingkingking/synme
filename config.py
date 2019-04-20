# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (129, 129, 129) 

# SSD300 CONFIGS
synme = { 
    'num_classes': 211, 
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1], 
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264], # min and max values of the object size to be predicted at each layer position.
    'max_sizes': [60, 111, 162, 213, 264, 315], 
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # 1 is default, so no need to add, please refer to prior_box 
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'synme',
}

DATASET_ROOT = os.path.join(HOME, 'data/synme/')
EVAL_DIR = os.path.join(DATASET_ROOT, 'eval')
DETECTION_DIR = os.path.join(DATASET_ROOT, 'detection')
