"""
Modified from PointNet++: https://github.com/charlesq34/pointnet2
Author: Wenxuan Wu
Date: July 2018
"""
import argparse
import math
import h5py
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
import provider
import tf_util
import scannet_dataset_rgb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
EPOCH_CNT_WHOLE = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BANDWIDTH = 0.05

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
Point_Util = os.path.join(BASE_DIR, 'utils', 'pointconv_util.py')
LOG_DIR = FLAGS.log_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (Point_Util, LOG_DIR))
os.system('cp %s %s' % ('PointConv.py', LOG_DIR))
os.system('cp train_scannet_IoU.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 21

# Shapenet official train/test split
DATA_PATH = os.path.join(BASE_DIR, 'scannet')
print("start loading training data ...")
TRAIN_DATASET = scannet_dataset_rgb.ScannetDataset(root=DATA_PATH, block_points=NUM_POINT, split='train')
print("start loading validation data ...")
TEST_DATASET = scannet_dataset_rgb.ScannetDataset(root=DATA_PATH, block_points=NUM_POINT, split='val')
print("start loading whole scene validation data ...")
TEST_DATASET_WHOLE_SCENE = scannet_dataset_rgb.ScannetDatasetWholeScene(root=DATA_PATH, block_points=NUM_POINT, split='val')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, BANDWIDTH, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        whole_test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'whole_scene'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        #sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            start_time = time.time()
            train_one_epoch(sess, ops, train_writer)
            end_time = time.time()
            log_string('one epoch time: %.4f'%(end_time - start_time))
            eval_one_epoch(sess, ops, test_writer)
            if epoch%5==0:
                acc = eval_whole_scene_one_epoch(sess, ops, whole_test_writer)
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = int(len(TRAIN_DATASET)/BATCH_SIZE)
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_iou_deno = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        #batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        batch_data, batch_label, batch_smpw = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation
        aug_data = provider.rotate_point_cloud_z(batch_data)
        #aug_data = provider.rotate_point_cloud(batch_data)

        feed_dict = {ops['pointclouds_pl']: aug_data,
                    ops['labels_pl']: batch_label,
                    ops['smpws_pl']:batch_smpw,
                    ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        iou_deno = 0
        for l in range(NUM_CLASSES):
            iou_deno += np.sum((pred_val==l) | (batch_label==l))
        total_iou_deno += iou_deno
        loss_sum += loss_val
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            log_string('total IoU: %f' % (total_correct / float(total_iou_deno)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_iou_deno = 0

# evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = int(len(TEST_DATASET)/BATCH_SIZE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    labelweights = np.zeros(21)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        aug_data = provider.rotate_point_cloud_z(batch_data)
        #aug_data = provider.rotate_point_cloud(batch_data)
        bandwidth = BANDWIDTH

        feed_dict = {ops['pointclouds_pl']: aug_data,
                    ops['labels_pl']: batch_label,
                    ops['smpws_pl']: batch_smpw,
                    ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label,range(22))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
            total_iou_deno_class[l] += np.sum(((pred_val==l) | (batch_label==l)) & (batch_smpw>0))

    mIoU = np.mean(np.array(total_correct_class[1:])/(np.array(total_iou_deno_class[1:],dtype=np.float)+1e-6))
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point avg class IoU: %f' % (mIoU))
    log_string('eval point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    iou_per_class_str = '------- IoU --------\n'
    for l in range(1,NUM_CLASSES):
        iou_per_class_str += 'class %d, acc: %f \n' % (l,total_correct_class[l]/float(total_iou_deno_class[l]))
    log_string(iou_per_class_str)
    EPOCH_CNT += 1
    return mIoU

# evaluate on whole scenes, for each block, only sample 8192 points
def eval_whole_scene_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT_WHOLE
    is_training = False
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----'%(EPOCH_CNT_WHOLE))

    labelweights = np.zeros(21)
    is_continue_batch = False
    
    extra_batch_data = np.zeros((0,NUM_POINT,3))
    extra_batch_label = np.zeros((0,NUM_POINT))
    extra_batch_smpw = np.zeros((0,NUM_POINT))
    for batch_idx in range(num_batches):
        if not is_continue_batch:
            batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,extra_batch_data),axis=0)
            batch_label = np.concatenate((batch_label,extra_batch_label),axis=0)
            batch_smpw = np.concatenate((batch_smpw,extra_batch_smpw),axis=0)
        else:
            batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,batch_data_tmp),axis=0)
            batch_label = np.concatenate((batch_label,batch_label_tmp),axis=0)
            batch_smpw = np.concatenate((batch_smpw,batch_smpw_tmp),axis=0)
        if batch_data.shape[0]<BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data.shape[0]==BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data = np.zeros((0,NUM_POINT,3))
            extra_batch_label = np.zeros((0,NUM_POINT))
            extra_batch_smpw = np.zeros((0,NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data = batch_data[BATCH_SIZE:,:,:]
            extra_batch_label = batch_label[BATCH_SIZE:,:]
            extra_batch_smpw = batch_smpw[BATCH_SIZE:,:]
            batch_data = batch_data[:BATCH_SIZE,:,:]
            batch_label = batch_label[:BATCH_SIZE,:]
            batch_smpw = batch_smpw[:BATCH_SIZE,:]

        aug_data = batch_data
        bandwidth = BANDWIDTH
        feed_dict = {ops['pointclouds_pl']: aug_data,
                    ops['labels_pl']: batch_label,
                    ops['smpws_pl']: batch_smpw,
                    ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label,range(22))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
            total_iou_deno_class[l] += np.sum(((pred_val==l) | (batch_label==l)) & (batch_smpw>0))

    mIoU = np.mean(np.array(total_correct_class[1:])/(np.array(total_iou_deno_class[1:],dtype=np.float)+1e-6))
    log_string('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point avg class IoU: %f' % (mIoU))
    log_string('eval whole scene point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))

    iou_per_class_str = '------- IoU --------\n'
    for l in range(1,NUM_CLASSES):
        iou_per_class_str += 'class %d, acc: %f \n' % (l,total_correct_class[l]/float(total_iou_deno_class[l]))
    log_string(iou_per_class_str)

    EPOCH_CNT_WHOLE += 1
    return mIoU


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
