import numpy as np

# Houston2013
Model = 'DMGO' 
FOLDER = "E:/datasets/"
DATASET = 'Houston2013'
TRAIN_SET = "E:/datasets/Houston2013/TRLabel.mat"
TEST_SET = "E:/datasets/Houston2013/TSLabel.mat"
EPOCH = 150
BATCH_SIZE = 64
PATCH_SIZE = 11
LR = 0.001
RESTORE = None

FLIP_AUGMENTATION = True
# hyper-parameters from OGM
LR_DECAY_RATIO = 0.4
LR_DECAY_STEP = 10
ALPHA = 0.4
MODULATION_STARTS = 20
MODULATION_ENDS = 70 
MODULATION = 'OGM' # [OGM, Normal]

