import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

class LW(Dataset):
    def __init__(self, root_dir, databse_file_path, query_file_path, transform=None, mode='train',
                 nNeg=5, posDistThr=0.5, negDistThr=25, cached_queries=1000, cached_negatives=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.nNeg = nNeg
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # Load image list
        self.qImages, self.qCoords = self.load_file(query_file_path)
        self.dbImages, self.dbCoords = self.load_file(databse_file_path)

        self.qIdx = np.arange(len(self.qImages))

        self.dbCoords = np.array(self.dbCoords)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qEndPosList = [len(self.qImages)]
        self.dbEndPosList = [len(self.dbImages)]

        self.build_ground_truth()

    def load_file(self, file_list):
        imgs = []
        coords = []
        with open(file_list, 'r') as f:
            for line in f:
                path, x, y, theta, *_ = line.strip().split()
                imgs.append(os.path.join(self.root_dir, path))
                coords.append([float(x), float(y)])
        return imgs, coords

    def build_ground_truth(self):
        print("Building ground truth for positives and negatives...")
        neigh = NearestNeighbors(radius=self.posDistThr)
        neigh.fit(self.dbCoords)
        self.pIdx = neigh.radius_neighbors(self.dbCoords, return_distance=False)
        self.all_pos_indices = self.pIdx  # for compatibility with val.py

        neigh_neg = NearestNeighbors(radius=self.negDistThr)
        neigh_neg.fit(self.dbCoords)
        all_neg = neigh_neg.radius_neighbors(self.dbCoords, return_distance=False)
        self.nonNegIdx = []
        for i in range(len(self.dbCoords)):
            negatives = np.setdiff1d(np.arange(len(self.dbCoords)), all_neg[i])
            self.nonNegIdx.append(negatives)


    def __getitem__(self, idx):
        triplet, _ = self.triplets[idx]
        qidx, pidx, *nidxs = triplet
    
        query = self.transform(Image.open(self.qImages[qidx]).convert('RGB'))
        positive = self.transform(Image.open(self.dbImages[pidx]).convert('RGB'))
        negatives = torch.stack([
            self.transform(Image.open(self.dbImages[n]).convert('RGB')) for n in nidxs
        ])
    
        return query, positive, negatives, [qidx, pidx] + nidxs

    def __len__(self):
        return len(self.qImages)

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None
        query, positive, negatives, indices = zip(*batch)
        query = torch.stack(query, 0)
        positive = torch.stack(positive, 0)
        negatives = torch.cat(negatives, 0)
        negCounts = torch.tensor([n.shape[0] for n in negatives])
        indices = [i for sublist in indices for i in sublist]
        return query, positive, negatives, negCounts, indices

    def new_epoch(self):
        self.triplets = []
        self.nCacheSubset = 1
        self.current_subset = 0

    def update_subcache(self, net=None, outputdim=None):
        self.triplets = []
    
        for q in self.qIdx:
            if len(self.pIdx[q]) == 0:
                continue
            qidx = q
            pidx = np.random.choice(self.pIdx[q])
            while True:
                nidxs = np.random.choice(self.nonNegIdx[q], self.nNeg, replace=False)
                if not any(n in self.pIdx[q] for n in nidxs):
                    break
    
            triplet = [qidx, pidx] + list(nidxs)
            target = [-1, 1] + [0] * self.nNeg
            self.triplets.append((triplet, target))
    
        self.current_subset += 1