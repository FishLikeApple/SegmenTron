DATASET:
    NAME: "cityscape"
    MEAN: [0.449]
    STD: [0.226]
TRAIN:
    EPOCHS: 1000
    BATCH_SIZE: 12
    CROP_SIZE: (512, 1024)
TEST:
    BATCH_SIZE: 4
    TEST_MODEL_PATH: 'runs/checkpoints/fast_scnn__cityscape_2019-11-19-02-02/best_model.pth'

SOLVER:
    LR: 0.045
    DECODER_LR_FACTOR: 1.0
    WEIGHT_DECAY: 4e-5
    AUX: True
    AUX_WEIGHT: 0.4

AUG:
    COLOR_JITTER: 0.4

MODEL:
    MODEL_NAME: "FastSCNN"
    BN_MOMENTUM: 0.01
