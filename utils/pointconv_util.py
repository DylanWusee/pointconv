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

def KDE(pts):
    '''
    pts: B * P * C
    density: B * P * 1
    '''
    with tf.variable_scope('KDE'):
        num_points = pts.get_shape()[1].value
        num_var = pts.get_shape()[-1].value
        dist = tf.norm(pts, axis = 2)
        _, variance = tf.nn.moments(dist, axes = 1)
        h_val = tf.multiply(tf.multiply(1.06, variance), float(num_points)**(-1.0/5))
        distance = batch_distance_matrix(pts, pts)
        h_scale = tf.expand_dims(tf.expand_dims(tf.multiply(h_val, h_val), axis = -1), axis = -1)
        distance = tf.divide(distance, h_scale)
        gauss_dist = tf.exp(tf.negative(distance))
        density = tf.reduce_sum(gauss_dist, axis = -1, keepdims = True)
        h_val2 = tf.multiply(float(num_points), h_val**num_var)
        h_scale2 = tf.expand_dims(tf.expand_dims(h_val2, axis = -1), axis = -1)
        density = tf.divide(density, h_scale2)
    return density


def kernel_density_estimation(pts, sigma, N_points = 128, radius = float('+Inf'), is_norm = False):
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

def grouping(feature, K, src_xyz, q_xyz, radius = 100.0, knn = True, use_xyz = True):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    knn: bool
    '''

    if knn:
        _, idx = tf_grouping.knn_point(K, src_xyz, q_xyz)
    else:
        idx, pts_cnt = tf_grouping.query_ball_point(radius, K, src_xyz, q_xyz)

    grouped_xyz = tf_grouping.group_point(src_xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(q_xyz, 2), [1, 1, K, 1])

    grouped_points = tf_grouping.group_point(feature, idx)
    if use_xyz:
        new_points = tf.concat([grouped_xyz, grouped_points], axis = -1)
    else:
        new_points = grouped_points
    
    return grouped_xyz, new_points

def batch_distance_matrix(A, B):
    '''
    input:
    A : B * N1 * C
    B : B * N2 * C
    output:
    distance : B * N1 * N2
    '''
    r_A = tf.reduce_sum(tf.multiply(A, A), axis = 2, keepdims = True)
    r_B = tf.reduce_sum(tf.multiply(B, B), axis = 2, keepdims = True)
    m = tf.matmul(A, tf.transpose(B, perm = (0, 2, 1)))
    distance = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return distance

def knn_indices(q_pts, pts, k, sort=True):
    '''
    input:
    q_pts: B * N1 * 3, queries points
    pts: B * N2 * 3, source points
    k : scalar, K nearest neighbors
    output:
    indices: B * N1 * k * 2 : B * N1 * k * (batch_id, point_id)
    '''
    batch_size = q_pts.get_shape()[0].value
    num_points = q_pts.get_shape()[1].value
    
    distance = batch_distance_matrix(q_pts, pts)
    distance, point_indices = tf.nn.top_k(tf.negative(distance), k= k, sorted=sort)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis = 3)], axis = 3)
    return -distance, indices

# indices is (B, N, K, 2)
# return shape is (B, N, K, 2)
def sort_points(points, indices, sorting_method):
    indices_shape = tf.shape(indices)
    batch_size = indices_shape[0]
    point_num = indices_shape[1]
    k = indices_shape[2]

    nn_pts = tf.gather_nd(points, indices)  # (N, P, K, 3)
    if sorting_method.startswith('c'):
        if ''.join(sorted(sorting_method[1:])) != 'xyz':
            print('Unknown sorting method!')
            exit()
        epsilon = 1e-8
        nn_pts_min = tf.reduce_min(nn_pts, axis=2, keep_dims=True)
        nn_pts_max = tf.reduce_max(nn_pts, axis=2, keep_dims=True)
        nn_pts_normalized = (nn_pts - nn_pts_min) / (nn_pts_max - nn_pts_min + epsilon)  # (N, P, K, 3)
        scaling_factors = [math.pow(100.0, 3 - sorting_method.find('x')),
                           math.pow(100.0, 3 - sorting_method.find('y')),
                           math.pow(100.0, 3 - sorting_method.find('z'))]
        scaling = tf.constant(scaling_factors, shape=(1, 1, 1, 3))
        sorting_data = tf.reduce_sum(nn_pts_normalized * scaling, axis=-1, keep_dims=False)  # (N, P, K)
    elif sorting_method == 'l2':
        nn_pts_center = tf.reduce_mean(nn_pts, axis=2, keep_dims=True)  # (N, P, 1, 3)
        nn_pts_local = tf.subtract(nn_pts, nn_pts_center)  # (N, P, K, 3)
        sorting_data = tf.norm(nn_pts_local, axis=-1, keep_dims=False)  # (N, P, K)
    else:
        print('Unknown sorting method!')
        exit()
    _, k_indices = tf.nn.top_k(sorting_data, k=k, sorted=True)  # (N, P, K)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
    point_indices = tf.tile(tf.reshape(tf.range(point_num), (1, -1, 1, 1)), (batch_size, 1, k, 1))
    k_indices_4d = tf.expand_dims(k_indices, axis=3)
    sorting_indices = tf.concat([batch_indices, point_indices, k_indices_4d], axis=3)  # (N, P, K, 3)
    return tf.gather_nd(indices, sorting_indices)






if __name__=='__main__':
    #test KDE

    pts = np.random.randn(1, 8192, 3).astype('float32')

    import pdb
    pdb.set_trace()

    with tf.device('/gpu:1'):
        points = tf.constant(pts)
        density = kernel_density_estimation(points, 1.0)

    with tf.Session('') as sess:
        den = sess.run(density)

    import scipy.io as sio 

    sio.savemat('density.mat', dict([('pts', pts), ('density', den)]))










