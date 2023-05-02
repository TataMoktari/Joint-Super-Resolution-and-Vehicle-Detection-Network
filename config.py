from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 1
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
#config.TRAIN.n_epoch_init = 100
config.TRAIN.n_epoch_init = 10
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 4
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 1)#2
config.TRAIN.initial_weight_srgan = "./checkpoint_128/SRGAN_weight_X4"


## train set location
config.TRAIN.hr_img_path = '/home/mousumi/PycharmProjects/Joint_Optimization_Fine_Tune_DOTA/JPEGImages_DOTA_512/'
config.TRAIN.lr_img_path = '/home/mousumi/PycharmProjects/Joint_Optimization_Fine_Tune_DOTA/test_dota_128/'


config.VALID = edict()
## test set location
config.VALID.hr_img_path = '/home/mousumi/PycharmProjects/Joint_Optimization_Fine_Tune_IR/JPEGImagesIR2RGB_512_Test'
config.VALID.lr_img_path = '/home/mousumi/PycharmProjects/Joint_Optimization_Fine_Tune_IR/JPEGImagesIR_128_Test'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")




