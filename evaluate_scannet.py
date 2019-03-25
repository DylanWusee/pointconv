"""
Evaluation on ScanNet: Generalize neccenary .ply and .txt file
Author: Wenxuan Wu
Date: July 2018
"""

import argparse
import math
from datetime import datetime
import h5py
from plyfile import PlyData, PlyElement
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'scannet'))
#sys.path.append(os.path.join(BASE_DIR, 'scannet/preprocessing'))
sys.path.append(os.path.join(BASE_DIR, 'scannet/visualize'))
import provider
import tf_util
import scannet_dataset_sw_rgb
import pc_util
from visualize_labels_on_mesh import visualize

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [256/512/1024/2048] [default: 8192]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--ply_path', default='scannet', help='ply path from original Scannet')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--num_votes', type=int, default=5, help='Aggregate classification scores from multiple rotations [default: 5]')
parser.add_argument('--with_rgb',help='With rgb or not', action='store_true')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
WITH_RGB = FLAGS.with_rgb
PLY_PATH = FLAGS.ply_path
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BANDWIDTH = 0.05

NUM_CLASSES = 21
HOSTNAME = socket.gethostname()

DATA_PATH = os.path.join(BASE_DIR, 'scannet')
print("start loading whole scene data ...")
TEST_DATASET_WHOLE_SCENE = scannet_dataset_sw_rgb.ScannetDatasetWholeScene_evaluation(root=DATA_PATH, split='val', with_rgb = WITH_RGB)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    with tf.device('/gpu:'+str(GPU_INDEX)):
        if WITH_RGB:
            pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 6))
        else:
            pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        labels_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_POINT))
        smpws_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, BANDWIDTH)
        MODEL.get_loss(pred, labels_pl, smpws_pl)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred}

    eval_one_epoch(sess, ops, num_votes)

def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

test_class = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False
    file_list = "./scannet/scannetv2_val.txt"
    with open(file_list) as fl:
        scene_id = fl.read().splitlines()
    
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EVALUATION WHOLE SCENE----')
    
    for batch_idx in range(num_batches):
        print("visualize %d %s ..."%(batch_idx, scene_id[batch_idx]))
        whole_scene_points_index = TEST_DATASET_WHOLE_SCENE.scene_points_id[batch_idx]
        whole_scene_points_num = TEST_DATASET_WHOLE_SCENE.scene_points_num[batch_idx]
        whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        for vote_idx in range(num_votes):
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            if WITH_RGB:
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
            else:
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))
            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1)*BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size,...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size,...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size,...] = scene_point_index[start_idx:end_idx, ...]

                if WITH_RGB:
                    batch_data[:, :, 3:6] /= 1.0 #255.0

                feed_dict = {ops['pointclouds_pl']: batch_data,
                        ops['labels_pl']: batch_label,
                        ops['is_training_pl']: is_training}
                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)#BxNxNUM_CLASSES
                batch_pred_label = np.argmax(pred_val[:, :, 1:], 2) + 1#BxN
                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size,...], batch_pred_label[0:real_batch_size,...])

        pred_label = np.argmax(vote_label_pool, 1)
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((whole_scene_label==l))
            total_correct_class[l] += np.sum((pred_label==l) & (whole_scene_label==l))
            total_iou_deno_class[l] += np.sum(((pred_label==l) | (whole_scene_label==l)) & (whole_scene_label > 0))


        print(total_correct_class)
        print(total_iou_deno_class)
        print(total_seen_class)
        whole_scene_data = np.zeros(whole_scene_points_num)
        whole_scene_data[whole_scene_points_index] = test_class[pred_label.astype(np.int32)]

        filename = os.path.join(DUMP_DIR, scene_id[batch_idx] + '.txt')
        with open(filename, 'w') as pl_save:
            for i in whole_scene_data:
                pl_save.write(str(int(i))+'\n')
            pl_save.close()
        
        pred_file = filename
        mesh_file = os.path.join(PLY_PATH, scene_id[batch_idx], scene_id[batch_idx]+ '_vh_clean_2.ply')
        output_file = os.path.join(DUMP_DIR, scene_id[batch_idx] + '.ply')
        visualize(pred_file, mesh_file, output_file)
        
    IoU = np.array(total_correct_class[1:])/(np.array(total_iou_deno_class[1:],dtype=np.float)+1e-6)
    log_string('eval point avg class IoU: %f' % (np.mean(IoU)))
    IoU_Class = 'Each Class IoU:::\n'
    for i in range(IoU.shape[0]):
        IoU_Class += 'Class %d : %.4f\n'%(i+1, IoU[i])
    log_string(IoU_Class)

    print("Done!")

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
