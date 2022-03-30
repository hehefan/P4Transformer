import os
import sys
import numpy as np
from torch.utils.data import Dataset

Cross_Subject = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

def clip_normalize(clip):
    pc = np.reshape(a=clip, newshape=[-1, 3])
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    clip = (clip - centroid) / m
    return clip

class NTU60Subject(Dataset):
    def __init__(self, root, meta, frames_per_clip=23, step_between_clips=2, num_points=2048, train=True):
        super(NTU60Subject, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0

        with open(meta, 'r') as f:
            for line in f:
                name, nframes = line.split()
                subject = int(name[9:12])
                if train:
                    if subject in Cross_Subject:
                        label = int(name[-3:]) - 1
                        nframes = int(nframes)
                        for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                            self.index_map.append((index, t))
                        index += 1
                        self.labels.append(label)
                        self.videos.append(os.path.join(root, name+'.npz'))
                else:
                    if subject not in Cross_Subject:
                        label = int(name[-3:]) - 1
                        nframes = int(nframes)
                        for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                            self.index_map.append((index, t))
                        index += 1
                        self.labels.append(label)
                        self.videos.append(os.path.join(root, name+'.npz'))

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        video = np.load(video, allow_pickle=True)['data'] * 100
        label = self.labels[index]

        clip = [video[t+i*self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = clip_normalize(np.array(clip))

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        return clip.astype(np.float32), label, index
