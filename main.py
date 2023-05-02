#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model_fine_tune_H import SRGAN_g_custom, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config

## Detector_model
import random
from core.config import cfg
import core.utils as utils
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from tqdm import tqdm
import cv2

# from model import YOLOV3


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
initial_weight_srgan = config.TRAIN.initial_weight_srgan
ni = int(np.sqrt(batch_size))

in_size = 128

###======================Detector=======================###
anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
classes = utils.read_class_names(cfg.YOLO.CLASSES)
num_classes = len(classes)
learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
learn_rate_end = cfg.TRAIN.LEARN_RATE_END
first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
initial_weight = cfg.TRAIN.INITIAL_WEIGHT
# self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
max_bbox_per_scale = 150

# self.train_logdir = "./data/log/train"
# trainset = Dataset('train')

annot_path = cfg.TRAIN.ANNOT_PATH
input_sizes = cfg.TRAIN.INPUT_SIZE
batch_size = cfg.TRAIN.BATCH_SIZE
data_aug = cfg.TRAIN.DATA_AUG
train_input_sizes = cfg.TRAIN.INPUT_SIZE
strides = np.array(cfg.YOLO.STRIDES)
classes = utils.read_class_names(cfg.YOLO.CLASSES)
anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
train_input_size = train_input_sizes
train_output_sizes = train_input_size // strides


def load_annotations(annot_path):
    with open("/home/moktari/PycharmProjects/Joint_Optimization_Fine_Tune_IR/data/dataset/voc_train.txt", 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt]


    return annotations


annotations = load_annotations(annot_path)
num_samples = len(annotations)
num_batchs = int(np.ceil(num_samples / batch_size))
batch_count = 0
steps_per_period = len(annotations)


def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes


def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes


def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes


def parse_annotation(annotation):
    line = annotation.split(' ')
    bboxes = np.array([list(map(int, box.split(','))) for box in line[0:]])

    bboxes = utils.image_preporcess([train_input_size, train_input_size], np.copy(bboxes))
    return bboxes


def bbox_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


def preprocess_true_boxes(bboxes):
    label = [np.zeros((train_output_sizes[i], train_output_sizes[i], anchor_per_scale,
                       5 + num_classes)) for i in range(3)]
    bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))

    for bbox in bboxes:
        bbox_coor = bbox[:4]
        bbox_class_ind = bbox[4]

        onehot = np.zeros(num_classes, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
        print(bbox_xywh_scaled, 'scaled')

        iou = []
        exist_positive = False
        for i in range(3):
            anchors_xywh = np.zeros((anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]

            iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                label[i][yind, xind, iou_mask, :] = 0
                label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                label[i][yind, xind, iou_mask, 4:5] = 1.0
                label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / anchor_per_scale)
            best_anchor = int(best_anchor_ind % anchor_per_scale)
            xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

            bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
    label_sbbox, label_mbbox, label_lbbox = label
    sbboxes, mbboxes, lbboxes = bboxes_xywh
    return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples_celeba/{}_ginit_{}".format(tl.global_flag['mode'], in_size)
    save_dir_gan = "samples_celeba/{}_gan_{}".format(tl.global_flag['mode'], in_size)
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint_{}".format(in_size)  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.jpg', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.jpg', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.jpg', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, in_size, in_size, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 256, 256, 3], name='t_target_image')
    t_target_image512 = tf.placeholder('float32', [batch_size, 512, 512, 3], name='t_target_image512')


    ## Detector_Placeholder

    detector_label_sbbox = tf.placeholder(dtype=tf.float32, name='sbbox_label')
    detector_label_mbbox = tf.placeholder(dtype=tf.float32, name='mbbox_label')
    detector_label_lbbox = tf.placeholder(dtype=tf.float32, name='lbbox_label')
    detector_true_sbboxes = tf.placeholder(dtype=tf.float32, name='detector_sbboxes')
    detector_true_mbboxes = tf.placeholder(dtype=tf.float32, name='detector_mbboxes')
    detector_true_lbboxes = tf.placeholder(dtype=tf.float32, name='detector_lbboxes')
    trainable = tf.placeholder(dtype=tf.bool, name='training')

    net_g, net_g512 = SRGAN_g_custom(t_image, is_train=True, reuse=False)

    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False, scope="SRGAN_d")
    net_d512, logits_real512 = SRGAN_d(t_target_image512, is_train=True, reuse=False, scope="SRGAN_d512")

    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True, scope="SRGAN_d")
    _, logits_fake512 = SRGAN_d(net_g512.outputs, is_train=True, reuse=True, scope="SRGAN_d512")

    ### Detector_model

    with tf.name_scope("define_loss"):
        model = YOLOV3(net_g512.outputs, trainable)
        giou_loss, conf_loss, prob_loss = model.compute_loss(detector_label_sbbox, detector_label_mbbox,
                                                             detector_label_lbbox, detector_true_sbboxes,
                                                             detector_true_mbboxes, detector_true_lbboxes)

    net_g.print_params(False)
    net_g.print_layers()
    net_g512.print_params(False)
    net_g512.print_layers()
    net_d.print_params(False)
    net_d.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)

    t_target_image_224_512 = tf.image.resize_images(t_target_image512, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224_512 = tf.image.resize_images(net_g512.outputs, size=[224, 224], method=0, align_corners=False)

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    _, vgg_target_emb512 = Vgg19_simple_api((t_target_image_224_512 + 1) / 2, reuse=True)
    _, vgg_predict_emb512 = Vgg19_simple_api((t_predict_image_224_512 + 1) / 2, reuse=True)

    ## test inference
    net_g_test_256, net_g_test_512 = SRGAN_g_custom(t_image, is_train=False, reuse=True)

    ####========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')

    d_loss3 = tl.cost.sigmoid_cross_entropy(logits_real512, tf.ones_like(logits_real512), name='d3')
    d_loss4 = tl.cost.sigmoid_cross_entropy(logits_fake512, tf.zeros_like(logits_fake512), name='d4')

    d_loss = d_loss3 + d_loss4 + d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_gan_loss512 = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake512, tf.ones_like(logits_fake512), name='g2')
    mse_loss512 = tl.cost.mean_squared_error(net_g512.outputs, t_target_image512, is_mean=True)
    vgg_loss512 = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb512.outputs, vgg_target_emb512.outputs, is_mean=True)

    ##Detector_Loss
    detector_loss = giou_loss + conf_loss + prob_loss
    weighted_detector_loss = 1e-3 * detector_loss

    ## Total_Loss
    g_loss = mse_loss + vgg_loss + g_gan_loss + mse_loss512 + vgg_loss512 + g_gan_loss512 + weighted_detector_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)
    d_vars512 = tl.layers.get_variables_with_name('SRGAN_d512', True, True)
    d_vars.extend(d_vars512)



    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss + mse_loss512, var_list=g_vars)

    ## SRGAN
    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)



    ###==========================SRGAN_Parameters_Pretrained=======================#####
    with tf.variable_scope("define_first_stage_train_srgan"):

        first_stage_optimizer_srgan = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([first_stage_optimizer_srgan]):
                train_op_with_frozen_variables_srgan = tf.no_op()

    with tf.variable_scope("define_second_stage_train_srgan"):
        second_stage_optimizer_srgan = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([second_stage_optimizer_srgan]):
                train_op_with_all_variables_srgan = tf.no_op()


    ###==========================Detector_Parameters_Pretrained========================####

    with tf.variable_scope('learn_rate'):
        global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        warmup_steps = tf.constant(warmup_periods * steps_per_period,
                                   dtype=tf.float64, name='warmup_steps')
        train_steps = tf.constant((first_stage_epochs + second_stage_epochs) * steps_per_period,
                                  dtype=tf.float64, name='train_steps')
        learn_rate = tf.cond(
            pred=global_step < warmup_steps,
            true_fn=lambda: global_step / warmup_steps * learn_rate_init,
            false_fn=lambda: learn_rate_end + 0.5 * (learn_rate_init - learn_rate_end) *
                             (1 + tf.cos(
                                 (global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
        )
        global_step_update = tf.assign_add(global_step, 1.0)

    with tf.variable_scope("define_first_stage_train"):
        detector_vars = []
        for var in tf.trainable_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                detector_vars.append(var)

        first_stage_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(detector_loss,var_list=detector_vars)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                train_op_with_frozen_variables = tf.no_op()


    with tf.variable_scope("define_second_stage_train"):
        second_stage_trainable_var_list = []

        for var in tf.global_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox', 'darknet', 'conv52', 'conv53', 'conv54','conv55', 'conv56', 'conv57', 'conv58', 'conv59', 'conv60', 'conv61', 'conv62',
                                    'conv63', 'conv64', 'conv65', 'conv66', 'conv67','conv68', 'conv_lobj_branch', 'conv_mobj_branch', 'conv_sobj_branch']:
                second_stage_trainable_var_list.append(var)

        second_stage_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(detector_loss, var_list=second_stage_trainable_var_list)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                train_op_with_all_variables = tf.no_op()




    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    sess.run(tf.variables_initializer(all_variables))

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")

    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    # sample_imgs = train_hr_imgs[0:batch_size]
    sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path,n_threads=32)  # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_256 = tl.prepro.threading_data(sample_imgs_384.copy(), fn=downsample_fn_3)
    sample_imgs_128 = tl.prepro.threading_data(sample_imgs_384.copy(), fn=downsample_fn_2)
    print('sample LR sub-image:', sample_imgs_256.shape, sample_imgs_256.min(), sample_imgs_256.max())
    print('sample LR sub-image:', sample_imgs_128.shape, sample_imgs_128.min(), sample_imgs_128.max())
    tl.vis.save_images(sample_imgs_256, [ni, ni], save_dir_ginit + '/_train_sample_256.png')
    tl.vis.save_images(sample_imgs_128, [ni, ni], save_dir_ginit + '/_train_sample_128.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_256, [ni, ni], save_dir_gan + '/_train_sample_256.png')
    tl.vis.save_images(sample_imgs_128, [ni, ni], save_dir_gan + '/_train_sample_128.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SRGAN_g'))
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        for idx in range(0, len(train_hr_img_list), batch_size):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx: idx + batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_128 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_2)
            b_imgs_256 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_3)
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init],
                               {t_image: b_imgs_128, t_target_image512: b_imgs_384, t_target_image: b_imgs_256})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1



        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
            epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 1 == 0):
            out, out512 = sess.run([net_g_test_256.outputs, net_g_test_512.outputs], {t_image: sample_imgs_128})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_256_%d.png' % epoch)
            tl.vis.save_images(out512, [ni, ni], save_dir_ginit + '/train_512_%d.png' % epoch)
            saver1.save(sess, checkpoint_dir + '/SRGAN_X4')

    ###========================= train GAN (SRGAN) =========================###
    loader = tf.train.Saver(second_stage_trainable_var_list)
    saver = tf.train.Saver(second_stage_trainable_var_list, max_to_keep=10)
    loader.restore(sess, initial_weight)

    loader1 = tf.train.Saver(g_vars)
    saver1 = tf.train.Saver(g_vars, max_to_keep=10)
    loader1.restore(sess, initial_weight_srgan)


    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        if epoch <= first_stage_epochs:
            train_op = train_op_with_frozen_variables
            train_op_srgan = train_op_with_frozen_variables_srgan
        else:
            train_op = train_op_with_all_variables
            train_op_srgan = train_op_with_all_variables_srgan

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        total_detector_loss = []
        for idx in range(0, len(train_hr_img_list), batch_size):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx:idx + batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_2)
            b_imgs_256 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_3)

            ##### Detector_Dataset##

            train_input_size = train_input_sizes
            train_output_sizes = train_input_size // strides

            batch_image = np.zeros((1, 128, 128, 3))

            batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0],
                                          anchor_per_scale, 5 + num_classes))
            batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1],
                                          anchor_per_scale, 5 + num_classes))
            batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2],
                                          anchor_per_scale, 5 + num_classes))

            batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))

            annotation = annotations[idx]
            bboxes = parse_annotation(annotation)
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(bboxes)

            batch_image[:, :, :, :] = b_imgs_96
            batch_label_sbbox[:, :, :, :, :] = label_sbbox
            batch_label_mbbox[:, :, :, :, :] = label_mbbox
            batch_label_lbbox[:, :, :, :, :] = label_lbbox
            batch_sbboxes[:, :, :] = sbboxes
            batch_mbboxes[:, :, :] = mbboxes
            batch_lbboxes[:, :, :] = lbboxes

            ## update D
            errD, _ = sess.run([d_loss, d_optim],
                               {t_image: b_imgs_96, t_target_image512: b_imgs_384, t_target_image: b_imgs_256})

            ## update_Detector

            _, errDetector, global_step_val = sess.run([train_op, detector_loss, global_step],
                                                       feed_dict={t_image: batch_image,
                                                                  detector_label_sbbox: batch_label_sbbox,
                                                                  detector_label_mbbox: batch_label_mbbox,
                                                                  detector_label_lbbox: batch_label_lbbox,
                                                                  detector_true_sbboxes: batch_sbboxes,
                                                                  detector_true_mbboxes: batch_mbboxes,
                                                                  detector_true_lbboxes: batch_lbboxes,
                                                                  trainable: True})
            total_detector_loss.append(errDetector)
            ## update G
            _, errG, errM, errV, errA, errM_512, errV_512, errA_512, errWeighted_detector_loss = sess.run([train_op_srgan, g_loss, mse_loss, vgg_loss, g_gan_loss, mse_loss512, vgg_loss512, g_gan_loss512,
                                                                                            weighted_detector_loss],
                                                                                            {t_image: b_imgs_96, t_target_image512: b_imgs_384,
                                                                                            t_target_image: b_imgs_256, detector_label_sbbox: batch_label_sbbox,
                                                                                            detector_label_mbbox: batch_label_mbbox,
                                                                                            detector_label_lbbox: batch_label_lbbox,
                                                                                            detector_true_sbboxes: batch_sbboxes,
                                                                                            detector_true_mbboxes: batch_mbboxes,
                                                                                            detector_true_lbboxes: batch_lbboxes,
                                                                                            trainable: True})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f detector_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f mse_loss512: %.6f, vgg_loss512: %.6f, g_gan_loss512: %.6f, weighted_detector_loss: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errDetector, errG, errM, errV, errA, errM_512,errV_512, errA_512, errWeighted_detector_loss))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1
        total_detector_loss = np.mean(total_detector_loss)

        log_detector = "[*] Epoch: [%2d/%2d] time: %4.4fs, detector_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_detector_loss)
        print(log_detector)
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
        print(log)


        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 1 == 0):
            out, out512 = sess.run([net_g_test_256.outputs, net_g_test_512.outputs], {t_image: sample_imgs_128})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_256_%d.png' % epoch)
            tl.vis.save_images(out512, [ni, ni], save_dir_gan + '/train_512_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 1 == 0):
            tl.files.save_npz(net_g512.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                              sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']),
                              sess=sess)
            saver1.save(sess, checkpoint_dir + '/SRGAN_X4')
            ckpt_file = "./checkpoint_detector/yolov3_train_loss=%.4f.ckpt" % total_detector_loss
            saver.save(sess, ckpt_file, global_step=epoch)

def evaluate():
    print('here 1')
    ## create folders to save result images
    save_dir = "samples/{}_{}".format(tl.global_flag['mode'], in_size)
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint_128"
    # checkpoint_dir = "/home/mousumi/PycharmProjects/Joint_Optimization_Fine Tune/checkpoint_128/best_weight"

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))

    #valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    #valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)


    ###========================== DEFINE MODEL ============================###
    for imid in range(len(train_lr_img_list)):
    #imid =0 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        filename = train_lr_img_list[imid]
        splitfilename = filename.split()
        splitfilename = splitfilename[0]
        train_lr_img = train_lr_imgs[imid]
        train_hr_img = train_hr_imgs[imid]
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        train_lr_img = (train_lr_img / 127.5) - 1  # rescale to ［－1, 1]

        size = train_lr_img.shape
        t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

        net_g, net_g512 = SRGAN_g_custom(t_image, is_train=False, reuse=False)

        ###========================== RESTORE G =============================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g512)

        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g512.outputs, {t_image: [train_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))

        print("LR size: %s /  generated HR size: %s" % (
        size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        path_fname = os.path.join(save_dir, splitfilename)
        tl.vis.save_image(out[0], path_fname)
        #tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
        #tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

        # out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 512 / in_size, size[1] * 512/ in_size],interp='bicubic', mode=None)
        # #out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 512 / in_size, size[1] * 512/ in_size],
        # tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
