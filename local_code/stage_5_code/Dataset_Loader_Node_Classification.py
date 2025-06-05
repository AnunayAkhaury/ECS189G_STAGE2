'''
Concrete IO class for a specific dataset
'''
import os

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD
import torch
import numpy as np
import scipy.sparse as sp

class Dataset_Loader(object):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__()

        # ─────────────────────────────────────────────────────────────────
        # Set a default path to your “stage_5_data” folder. You can override this later.
        PROJECT_ROOT = "/Users/anunayakhaury/Personal/UCDavis/ECS189G/Project/ECS189G_STAGE2"
        # Expect sub‐folders at: PROJECT_ROOT/data/stage_5_data/cora (or citeseer, pubmed)
        self.dataset_source_folder_path = os.path.join(
            PROJECT_ROOT, "data", "stage_5_data", dName
        )
        self.dataset_name = dName
        self.dataset_description = dDescription
        self.seed = seed
        # ─────────────────────────────────────────────────────────────────

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        idx_val = None
        if self.dataset_name == 'cora':
            idx_train = range(140)
            idx_val   = range(200, 500)
            idx_test  = range(500, 1500)


        elif self.dataset_name == "citeseer":

            labels_np = labels.numpy()

            num_classes = int(labels.max().item()) + 1

            rng = np.random.RandomState(self.seed)

            all_indices = np.arange(labels_np.shape[0])

            # 1) pick 20 nodes per class → 120 total train

            train_list = []

            for cls in range(num_classes):
                cls_nodes = np.where(labels_np == cls)[0]

                rng.shuffle(cls_nodes)

                train_list.append(cls_nodes[:20])

            train_idx = np.hstack(train_list)

            # 2) pool = everything except the 120 train nodes

            pool = np.setdiff1d(all_indices, train_idx, assume_unique=True)

            rng.shuffle(pool)

            # 3) from pool: 500 for val, then 1200 for test

            val_idx = pool[:500]

            test_idx = pool[500:500 + 1200]

            idx_train = train_idx

            idx_val = val_idx

            idx_test = test_idx

        elif self.dataset_name == "pubmed":
            labels_np = labels.numpy()
            num_classes = int(labels.max().item()) + 1
            rng = np.random.RandomState(self.seed)
            all_indices = np.arange(labels_np.shape[0])

                # 1) pick 20 nodes per class for train (60 total)
            train_list = []
            for cls in range(num_classes):
                cls_nodes = np.where(labels_np == cls)[0]
                rng.shuffle(cls_nodes)
                train_list.append(cls_nodes[:20])
            train_idx = np.hstack(train_list)

                # 2) remove training nodes from pool
            pool = np.setdiff1d(all_indices, train_idx, assume_unique=True)
            rng.shuffle(pool)

                # 3) from the pool, take next 500 for val, next 600 for test
            val_idx = pool[:500]
            test_idx = pool[500:1100]

            idx_train = torch.LongTensor(train_idx.tolist())
            idx_val = torch.LongTensor(val_idx.tolist())
            idx_test = torch.LongTensor(test_idx.tolist())

        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        # get the training nodes/testing nodes
        train_x = features[idx_train]
        val_x = features[idx_val]
        test_x = features[idx_test]
        print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels,
                 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}
