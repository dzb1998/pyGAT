import numpy as np
import scipy.sparse as sp
import torch

import csv
import unicodedata
import os



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    with open('data/kg/entities.dict') as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    row_1 = []
    row_2 = []

    with open('data/kg/output.csv') as fin:
        for row in csv.reader(fin, delimiter=','):
            if row[0] == '':
                continue
            # print(row)
            row_1_ = entity2id[row[1]]
            row_2_ = row[2].strip('][').split(', ')
            # print(row_2)
            row_2_ = [float(i) for i in row_2_]
            row_1.append(row_1_)
            row_2.append(row_2_)
            # print(row_1)
            # print(row_2)

    # print(row_1)
    # print(row_2)

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # print(idx)
    idx_ = np.array(row_1, dtype=np.int32)
    # print(idx_)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # print(idx_map)
    # idx_map_ = {j: i for i, j in enumerate(idx_)}
    # print(idx_map_)
    idx_map_ = {}
    for i, j in enumerate(idx_):
        idx_map_[j] = i
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges_unordered = np.genfromtxt('data/kg/train_nodes.txt', dtype=np.int32)
    # print(edges_unordered)
    edges = np.array(list(map(idx_map_.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    # print(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(idx_.shape[0], idx_.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(np.array(row_2))
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    # features = torch.FloatTensor(np.array(features.todense()))
    features = torch.FloatTensor(features)
    # labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # return adj, features, labels, idx_train, idx_val, idx_test
    return adj, features, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

