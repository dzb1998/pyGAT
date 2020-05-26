from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

from torch.utils.data import (DataLoader,
                             RandomSampler,
                             SequentialSampler,
                             TensorDataset)
from tqdm import tqdm, trange

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj, features, idx_train, idx_val, idx_test, edges_head, edges_tail = load_data()

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                # nclass=int(labels.max()) + 1, 
                nclass = 128,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                # nclass=int(labels.max()) + 1, 
                nclass = 128,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    # labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    edges_head = edges_head.cuda()
    edges_tail = edges_tail.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)
features, adj = Variable(features), Variable(adj)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


def pred_loss():
    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

    if args.cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

    negative_score = model((positive_sample, negative_sample), False, mode=mode)

    # if args.negative_adversarial_sampling:
    #     #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
    #     #print(negative_score, args.adversarial_temperature)
    #     negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
    #                       * F.logsigmoid(-negative_score)).sum(dim = 1)
    # else:
    negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

    positive_score = model(positive_sample)

    positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

    if args.uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

    loss = (positive_sample_loss + negative_sample_loss)/2
    return loss


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

batchsize = 4
tensor_dataset = TensorDataset(edges_head, edges_tail)
data_loader = DataLoader(tensor_dataset, batch_size = batchsize, shuffle = True)

for epoch in range(args.epochs):
    model.train()

    for batch in tqdm(data_loader):
        head, tail  = tuple(t for t in batch)
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features[head]+features[tail], adj)
        head_emb = output[0]
        tail_emb = output[1]

        neg_idx = torch.randint(0, 7127, (4, 2))
        neg_head_emb = output[neg_idx[:, 0]]
        neg_tail_emb = output[neg_idx[:, 1]]

        # loss_train = pred_loss(head_emb, tail_emb)
        posi = -F.logsigmoid(-torch.mm(torch.t(head_emb), tail_emb))
        nega = F.logsigmoid(torch.mm(torch.t(neg_head_emb), neg_tail_emb))
        loss_train = torch.sum(posi - nega)

        loss_train.backward()
        optimizer.step()

        print(loss_train)

#     loss_values.append(train(epoch))

#     torch.save(model.state_dict(), '{}.pkl'.format(epoch))
#     if loss_values[-1] < best:
#         best = loss_values[-1]
#         best_epoch = epoch
#         bad_counter = 0
#     else:
#         bad_counter += 1

#     if bad_counter == args.patience:
#         break

#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)

# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)

# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# # Testing
# compute_test()
