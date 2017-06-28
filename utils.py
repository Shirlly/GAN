import numpy as np
from math import ceil


def group_mse(ypreds, ys):
    scores = []
    for ypred, y in zip(ypreds, ys):
        scores.append(np.mean((ypred-y)**2))
    return scores

def group_accuracy(ypreds, ys):
    scores = []
    for ypred, y in zip(ypreds, ys):
        ypred = np.argmax(ypred, axis=1)
        y = np.argmax(y, axis=1)
        scores.append(np.sum(ypred == y)/float(len(y)))
    return scores


def total_mse(ypreds, ys):
    scores = []
    for ypred, y in zip(ypreds, ys):
        mse = (ypred-y)**2
        scores.append(np.sum(np.mean(mse,axis=1)))
    return scores

def total_accuracy(ypreds, ys):
    scores = []
    for ypred, y in zip(ypreds, ys):
        # import pdb; pdb.set_trace()
        ypred = np.argmax(ypred, axis=1)
        y = np.argmax(y, axis=1)
        scores.append(np.sum(ypred == y))
    return scores


def same(in_height, in_width, strides, filters):
    out_height = ceil(float(in_height) / float(strides[0]))
    out_width  = ceil(float(in_width) / float(strides[1]))
    return out_height, out_width

def valid(in_height, in_width, strides, filters):
    out_height = ceil(float(in_height - filters[0] + 1) / float(strides[0]))
    out_width  = ceil(float(in_width - filters[1] + 1) / float(strides[1]))
    return out_height, out_width
