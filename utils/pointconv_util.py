"""
Helper Function for PointConv
Author: Wenxuan Wu
Date: July 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np
import tensorflow as tf
from transforms3d.euler import euler2mat
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/grouping'))
import tf_sampling
import tf_grouping
from sklearn.neighbors import KDTree

def knn_kdtree(nsample, xyz, new_xyz):
    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    indices = np.zeros((batch_size, n_points, nsample), dtype=np.int32)
    for batch_idx in range(batch_size):
        X = xyz[batch_idx, ...]
        q_X = new_xyz[batch_idx, ...]
        kdt = KDTree(X, leaf_size=30)
        _, indices[batch_idx] = kdt.query(q_X, k = nsample)

    return indices

def kernel_density_estimation_ball(pts, radius, sigma, N_points = 128, is_norm = False):
    with tf.variable_scope("ComputeDensity") as sc:
        idx, pts_cnt = tf_grouping.query_ball_point(radius, N_points, pts, pts)
        g_pts = tf_grouping.group_point(pts, idx)
        g_pts -= tf.tile(tf.expand_dims(pts, 2), [1, 1, N_points, 1])

        R = tf.sqrt(sigma)
        xRinv = tf.div(g_pts, R)
        quadform = tf.reduce_sum(tf.square(xRinv), axis = -1)
        logsqrtdetSigma = tf.log(R) * 3
        mvnpdf = tf.exp(-0.5 * quadform - logsqrtdetSigma - 3 * tf.log(2 * 3.1415926) / 2)

        first_val, _ = tf.split(mvnpdf, [1, N_points - 1], axis = 2)

        mvnpdf = tf.reduce_sum(mvnpdf, axis = 2, keepdims = True)

        num_val_to_sub = tf.expand_dims(tf.cast(tf.subtract(N_points, pts_cnt), dtype = tf.float32), axis = -1)

        val_to_sub = tf.multiply(first_val, num_val_to_sub)

        mvnpdf = tf.subtract(mvnpdf, val_to_sub)

        scale = tf.div(1.0, tf.expand_dims(tf.cast(pts_cnt, dtype = tf.float32), axis = -1))
        density = tf.multiply(mvnpdf, scale)

        if is_norm:
            #grouped_xyz_sum = tf.reduce_sum(grouped_xyz, axis = 1, keepdims = True)
            density_max = tf.reduce_max(density, axis = 1, keepdims = True)
            density = tf.div(density, density_max)

        return density

def kernel_density_estimation(pts, sigma, kpoint = 32, is_norm = False):
    with tf.variable_scope("ComputeDensity") as sc:
        batch_size = pts.get_shape()[0]
        num_points = pts.get_shape()[1]
        if num_points < kpoint:
            kpoint = num_points.value - 1
        with tf.device('/cpu:0'):
            point_indices = tf.py_func(knn_kdtree, [kpoint, pts, pts], tf.int32)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, kpoint, 1))
        idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis = 3)], axis = 3)
        idx.set_shape([batch_size, num_points, kpoint, 2])

        grouped_pts = tf.gather_nd(pts, idx)
        grouped_pts -= tf.tile(tf.expand_dims(pts, 2), [1,1,kpoint,1]) # translation normalization

        R = tf.sqrt(sigma)
        xRinv = tf.div(grouped_pts, R)
        quadform = tf.reduce_sum(tf.square(xRinv), axis = -1)
        logsqrtdetSigma = tf.log(R) * 3
        mvnpdf = tf.exp(-0.5 * quadform - logsqrtdetSigma - 3 * tf.log(2 * 3.1415926) / 2)
        mvnpdf = tf.reduce_sum(mvnpdf, axis = 2, keepdims = True)

        scale = 1.0 / kpoint
        density = tf.multiply(mvnpdf, scale)

        if is_norm:
            #grouped_xyz_sum = tf.reduce_sum(grouped_xyz, axis = 1, keepdims = True)
            density_max = tf.reduce_max(density, axis = 1, keepdims = True)
            density = tf.div(density, density_max)

        return density

def sampling(npoint, pts):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * 3, input point cloud
    output:
    sub_pts: B * npoint * 3, sub-sampled point cloud
    '''

    sub_pts = tf_sampling.gather_point(pts, tf_sampling.farthest_point_sample(npoint, pts))
    return sub_pts

def grouping(feature, K, src_xyz, q_xyz, use_xyz = True):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''

    batch_size = src_xyz.get_shape()[0]
    npoint = q_xyz.get_shape()[1]

    point_indices = tf.py_func(knn_kdtree, [K, src_xyz, q_xyz], tf.int32)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
    idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis = 3)], axis = 3)
    idx.set_shape([batch_size, npoint, K, 2])

    grouped_xyz = tf.gather_nd(src_xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(q_xyz, 2), [1,1,K,1]) # translation normalization

    grouped_feature = tf.gather_nd(feature, idx)
    if use_xyz:
        new_points = tf.concat([grouped_xyz, grouped_feature], axis = -1)
    else:
        new_points = grouped_feature
    
    return grouped_xyz, new_points, idx

if __name__=='__main__':
    #test KDE
    import time
    batch_size = 8
    num_point = 8192
    pts = np.random.randn(batch_size, num_point, 3).astype('float32')

    import pdb
    pdb.set_trace()

    with tf.device('/gpu:1'):
        points = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
        density = kernel_density_estimation_ball(points, 1.0)
        #density = kernel_density_estimation(points, 1.0)
    init = tf.global_variables_initializer()
    with tf.Session('') as sess:
                
        sess.run(init)
        t1 = time.time()
        den = sess.run(density, feed_dict = {points:pts})

    print(time.time() - t1)

    #import scipy.io as sio 

    #sio.savemat('density.mat', dict([('pts', pts), ('density', den)]))










