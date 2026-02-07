# config.py

# model
TEACHER_MODEL = "resnet34"
STUDENT_MODEL = "resnet18"
MODEL_NAME = "resnet18"   # "resnet50" or "resnet101"

NUM_CLASSES = 10
KD_LAYER = "layer3"

# training
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LAMBDA_KD = 1.0

# system
NUM_WORKERS = 4
DEVICE = "cuda"

# checkpoint
SAVE_DIR = "./checkpoints_kd"
SAVE_LAST_NAME = "student_last.pth"
SAVE_BEST_NAME = "student_best.pth"

teacher_load_from = "./checkpoints/resnet34_epoch10.pth"
