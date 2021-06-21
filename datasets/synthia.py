import os
import sys
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

index_to_label = np.array([12, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1, 11], dtype='int32')
label_to_index = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 0], dtype='int32')
index_to_class = ['Void', 'Sky', 'Building', 'Road', 'Sidewalk', 'Fence', 'Vegetation', 'Pole', 'Car', 'Traffic Sign', 'Pedestrian', 'Bicycle', 'Lanemarking', 'Reserved', 'Reserved', 'Traffic Light']

def index_to_label_func(x):
    return index_to_label[x]
index_to_label_vec_func = np.vectorize(index_to_label_func)

class SegDataset(Dataset):
    def __init__(self, root='data/pc', meta='data/train_raw.txt', labelweight = 'data/labelweights.npz', frames_per_clip=3, num_points=16384, train=True):
        super(SegDataset, self).__init__()

        self.num_points = num_points
        self.train = train
        self.root = root
        self.frames_per_clip = frames_per_clip

        labelweights = np.load(labelweight)['labelweights'].astype(np.float32)
        if train:
            labelweights = 1/np.log(1.2 + labelweights)
            self.labelweights = labelweights / labelweights.min()
        else:
            self.labelweights = np.ones_like(labelweights)

        self.meta = []
        self.data = {}
        #m = 0
        with open(meta, 'r') as f:
            for line in f:
                line = line.split(' ')[0]
                line = line.split('/')
                sequence_name = line[0]
                frame_id = int(line[-1].split('.')[0])

                fn = os.path.join(root, sequence_name + '-' + str(frame_id).zfill(6) + '.npz')
                data = np.load(fn)

                pc = data['pc']             # (16384, 3)
                rgb = data['rgb']           # (16384, 3)
                semantic = data['semantic'] # (16384, )
                center = data['center']     # (3, )
                semantic = semantic.astype('uint8')

                self.data[sequence_name + '-' + str(frame_id)] = (pc, rgb, semantic, center)
                self.meta.append([sequence_name, frame_id])
                #m+=1
                #if m == 100:
                    #break
        self.meta.sort()

    def __len__(self):
        return len(self.meta)

    def read_training_data_point(self, index):
        sequence_name, frame_id = self.meta[index]

        pcs = []
        rgbs = []
        semantics = []
        center_0 = None

        most_recent_success = -1
        for diff in range(0, self.frames_per_clip):
            key = sequence_name + '-' + str(frame_id-diff)
            if key in self.data:
                pc, rgb, semantic, center = self.data[key]
                most_recent_success = frame_id - diff
            else:
                pc, rgb, semantic, center = self.data[sequence_name + '-' + str(most_recent_success)]

            if diff == 0:
                center_0 = center

            pcs.append(pc)
            rgbs.append(rgb)
            semantics.append(semantic)

        pc = np.stack(pcs, axis=0)
        rgb = np.stack(rgbs, axis=0)
        semantic = np.stack(semantics, axis=0)

        return pc, rgb, semantic, center_0


    def half_crop_w_context(self, half, context, pc, rgb, semantic, center):
        frames_per_clip = pc.shape[0]
        all_idx = np.arange(pc.shape[1])
        sample_indicies_half_w_context = []
        if half == 0:
            for f in range(frames_per_clip):
                sample_idx_half_w_context = all_idx[pc[f, :, 2] > (center[2] - context)]
                sample_indicies_half_w_context.append(sample_idx_half_w_context)
        else:
            for f in range(frames_per_clip):
                sample_idx_half_w_context = all_idx[pc[f, :, 2] < (center[2] + context)]
                sample_indicies_half_w_context.append(sample_idx_half_w_context)

        pc_half_w_context = [pc[f, s] for f, s in enumerate(sample_indicies_half_w_context)]
        rgb_half_w_context = [rgb[f, s] for f, s in enumerate(sample_indicies_half_w_context)]
        semantic_half_w_context = [semantic[f, s] for f, s in enumerate(sample_indicies_half_w_context)]
        if half == 0:
            loss_masks = [p[:, 2] > center[2] for p in pc_half_w_context]
        else:
            loss_masks = [p[:, 2] < center[2] for p in pc_half_w_context]
        valid_pred_idx_in_full = sample_indicies_half_w_context

        return pc_half_w_context, rgb_half_w_context, semantic_half_w_context, loss_masks, valid_pred_idx_in_full

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center

        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * 2)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center
        return pc

    def mask_and_label_conversion(self, semantic, loss_mask):
        labels = []
        loss_masks = []
        for i, s in enumerate(semantic):
            sem = s.astype('int32')
            label = index_to_label_vec_func(sem)
            loss_mask_ = (label != 12) * loss_mask[i]
            label[label == 12] = 0

            labels.append(label)
            loss_masks.append(loss_mask_)
        return labels, loss_masks

    def choice_to_num_points(self, pc, rgb, label, loss_mask, valid_pred_idx_in_full):

        # shuffle idx to change point order (change FPS behavior)
        for f in range(self.frames_per_clip):
            idx = np.arange(pc[f].shape[0])
            choice_num = self.num_points
            if pc[f].shape[0] > choice_num:
                shuffle_idx = np.random.choice(idx, choice_num, replace=False)
            else:
                shuffle_idx = np.concatenate([np.random.choice(idx, choice_num -  idx.shape[0]), np.arange(idx.shape[0])])
            pc[f] = pc[f][shuffle_idx]
            rgb[f] = rgb[f][shuffle_idx]
            label[f] = label[f][shuffle_idx]
            loss_mask[f] = loss_mask[f][shuffle_idx]
            valid_pred_idx_in_full[f] = valid_pred_idx_in_full[f][shuffle_idx]

        pc = np.stack(pc, axis=0)
        rgb = np.stack(rgb, axis=0)
        label = np.stack(label, axis=0)
        loss_mask = np.stack(loss_mask, axis=0)
        valid_pred_idx_in_full = np.stack(valid_pred_idx_in_full, axis=0)

        return pc, rgb, label, loss_mask, valid_pred_idx_in_full

    def __getitem__(self, index):
        context = 1.

        pc, rgb, semantic, center = self.read_training_data_point(index)

        half = 0
        pc1, rgb1, semantic1, mask1, valid_pred_idx_in_full1 = self.half_crop_w_context(half, context, pc, rgb, semantic, center)
        label1, mask1 = self.mask_and_label_conversion(semantic1, mask1)
        pc1, rgb1, label1, mask1, valid_pred_idx_in_full1 = self.choice_to_num_points(pc1, rgb1, label1, mask1, valid_pred_idx_in_full1)

        half = 1
        pc2, rgb2, semantic2, mask2, valid_pred_idx_in_full2 = self.half_crop_w_context(half, context, pc, rgb, semantic, center)
        label2, mask2 = self.mask_and_label_conversion(semantic2, mask2)
        pc2, rgb2, label2, mask2, valid_pred_idx_in_full2 = self.choice_to_num_points(pc2, rgb2, label2, mask2, valid_pred_idx_in_full2)

        if self.train:
            pc1 = self.augment(pc1, center)
            pc2 = self.augment(pc2, center)

        rgb1 = np.swapaxes(rgb1, 1, 2)
        rgb2 = np.swapaxes(rgb2, 1, 2)

        return pc1.astype(np.float32), rgb1.astype(np.float32), label1.astype(np.int64), mask1.astype(np.float32), pc2.astype(np.float32), rgb2.astype(np.float32), label2.astype(np.int64), mask2.astype(np.float32)

