""" ScanNet Class From Charles R. Qi, Hao Su.
Modiyied to support point-wise evaluation in ScanNet v2.
Author: Wenxuan Wu
Date: July 2018
"""


import pickle
import os
import sys
import numpy as np

class ScannetDatasetWholeScene_evaluation():
    #prepare to give prediction on each points
    def __init__(self, root, split='test', num_class = 21, block_points = 8192, with_rgb = True):
        self.root = root
        self.split = split
        self.with_rgb = with_rgb
        self.block_points = block_points
        self.point_num = []
        self.data_filename = os.path.join(self.root, 'scannet_%s_rgb21c_pointid.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
            self.scene_points_id = pickle.load(fp)
            self.scene_points_num = pickle.load(fp)
        if split=='train':
            labelweights = np.zeros(num_class)
            for seg in self.semantic_labels_list:
                self.point_num.append(seg.shape[0])
                tmp,_ = np.histogram(seg,range(num_class+1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            #self.labelweights = 1/np.log(1.2+labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1/3.0)
        else:
            self.labelweights = np.ones(num_class)
            for seg in self.semantic_labels_list:
                self.point_num.append(seg.shape[0])
    
    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    def split_data(self, data, idx):
        new_data = []
        for i in range(len(idx)):
            new_data += [np.expand_dims(data[idx[i]], axis = 0)]
        return new_data
    
    def nearest_dist(self, block_center, block_center_list):
        num_blocks = len(block_center_list)
        dist = np.zeros(num_blocks)
        for i in range(num_blocks):
            dist[i] = np.linalg.norm(block_center_list[i] - block_center, ord = 2) #i->j
        return np.argsort(dist)[0]

    def __getitem__(self, index):
        delta = 0.5
        if self.with_rgb:
            point_set_ini = self.scene_points_list[index]
        else:
            point_set_ini = self.scene_points_list[index][:, 0:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini[:, 0:3],axis=0)
        coordmin = np.min(point_set_ini[:, 0:3],axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/delta).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/delta).astype(np.int32)
        point_sets = []
        semantic_segs = []
        sample_weights = []
        point_idxs = []
        block_center = []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*delta,j*delta,0]
                curmax = curmin+[1.5,1.5,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini[:,0:3]>=(curmin-0.2))*(point_set_ini[:,0:3]<=(curmax+0.2)),axis=1)==3
                curchoice_idx = np.where(curchoice)[0]
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.001))*(cur_point_set[:,0:3]<=(curmax+0.001)),axis=1)==3
                sample_weight = self.labelweights[cur_semantic_seg]
                sample_weight *= mask # N
                point_sets.append(cur_point_set) # 1xNx3/6
                semantic_segs.append(cur_semantic_seg) # 1xN
                sample_weights.append(sample_weight) # 1xN
                point_idxs.append(curchoice_idx) #1xN
                block_center.append((curmin[0:2] + curmax[0:2]) / 2.0)

        # merge small blocks
        num_blocks = len(point_sets)
        block_idx = 0
        while block_idx < num_blocks:
            if point_sets[block_idx].shape[0] > 4096:
                block_idx += 1
                continue
            
            small_block_data = point_sets[block_idx].copy()
            small_block_seg = semantic_segs[block_idx].copy()
            small_block_smpw = sample_weights[block_idx].copy()
            small_block_idxs = point_idxs[block_idx].copy()
            small_block_center = block_center[block_idx].copy()
            point_sets.pop(block_idx)
            semantic_segs.pop(block_idx)
            sample_weights.pop(block_idx)
            point_idxs.pop(block_idx)
            block_center.pop(block_idx)
            nearest_block_idx = self.nearest_dist(small_block_center, block_center)
            point_sets[nearest_block_idx] = np.concatenate((point_sets[nearest_block_idx], small_block_data), axis = 0)
            semantic_segs[nearest_block_idx] = np.concatenate((semantic_segs[nearest_block_idx], small_block_seg), axis = 0)
            sample_weights[nearest_block_idx] = np.concatenate((sample_weights[nearest_block_idx], small_block_smpw), axis = 0)
            point_idxs[nearest_block_idx] = np.concatenate((point_idxs[nearest_block_idx], small_block_idxs), axis = 0)
            num_blocks = len(point_sets)

        #divide large blocks
        num_blocks = len(point_sets)
        div_blocks = []
        div_blocks_seg = []
        div_blocks_smpw = []
        div_blocks_idxs = []
        div_blocks_center = []
        for block_idx in range(num_blocks):
            cur_num_pts = point_sets[block_idx].shape[0]

            point_idx_block = np.array([x for x in range(cur_num_pts)])
            if point_idx_block.shape[0]%self.block_points != 0:
                makeup_num = self.block_points - point_idx_block.shape[0]%self.block_points
                np.random.shuffle(point_idx_block)
                point_idx_block = np.concatenate((point_idx_block,point_idx_block[0:makeup_num].copy()))

            np.random.shuffle(point_idx_block)

            sub_blocks = list(self.chunks(point_idx_block, self.block_points))

            div_blocks += self.split_data(point_sets[block_idx], sub_blocks)
            div_blocks_seg += self.split_data(semantic_segs[block_idx], sub_blocks)
            div_blocks_smpw += self.split_data(sample_weights[block_idx], sub_blocks)
            div_blocks_idxs += self.split_data(point_idxs[block_idx], sub_blocks)
            div_blocks_center += [block_center[block_idx].copy() for i in range(len(sub_blocks))]
        div_blocks = np.concatenate(tuple(div_blocks),axis=0)
        div_blocks_seg = np.concatenate(tuple(div_blocks_seg),axis=0)
        div_blocks_smpw = np.concatenate(tuple(div_blocks_smpw),axis=0)
        div_blocks_idxs = np.concatenate(tuple(div_blocks_idxs),axis=0)
        return div_blocks, div_blocks_seg, div_blocks_smpw, div_blocks_idxs
    def __len__(self):
        return len(self.scene_points_list)

if __name__=='__main__':
    import pdb
    pdb.set_trace()
    #d = ScannetDataset(root = '../data/scannet/scannet_v2', split='test', npoints=8192)
    d = ScannetDatasetWholeScene_evaluation(root = './data_v2')
    labelweights_vox = np.zeros(21)
    for ii in range(len(d)):
        print(ii)
        ps,seg,smpw, idxs = d[ii]
    print(labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32)))
    exit()

