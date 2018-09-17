import sys
import os
import re
import math
import numpy as np
import random

class BatchStrategy(object):
    def __init__(self, graph_mat, params):
        self.graph = np.array(graph_mat)
        self.n = 0
        self.m = 0
        self.num_nodes = len(graph_mat)
        self.pre_sum = np.zeros([self.num_nodes, self.num_nodes])

        if "contain_self" in params and params["contain_self"] is False:
            for i in xrange(num_nodes):
                self.graph[i][i] = 0.0
            self.graph = self.graph / np.sum(self.graph, axis = 1, keepdims = True)

        self.num_skips = params["num_skips"] if "num_skips" in params else 2

        for i in xrange(self.num_nodes):
            for j in xrange(self.num_nodes):
                self.pre_sum[i][j] = graph_mat[i][j]
                if j > 0:
                    self.pre_sum[i][j] += self.pre_sum[i][j - 1]

    def sequential_weighted(self, batch_size, **dictArg):
        batch_x = []
        batch_y = []
        for _ in xrange(batch_size):
            batch_x.append([self.n])
            batch_y.append(self.graph[self.n])
            self.n = (self.n + 1) % self.num_nodes
        return batch_x, batch_y

    @staticmethod
    def sampling_from_prob_array(a, pre_sum):
        x = random.random()
        return a[np.searchsorted(pre_sum, x)]

    def weighted_sampling(self, batch_size, **dictArg):
        batch_x = []
        batch_y = []
        for _ in xrange(batch_size):
            batch_x.append(self.n)
            batch_y.append([])
            for i in xrange(dictArg["neighbor_size"]):
                batch_y[-1].append(BatchStrategy.sampling_from_prob_array(self.graph[self.n], self.pre_sum[self.n]))
            self.m = (self.m + 1) % self.num_skips
            if self.m == 0:
                self.n = (self.n + 1) % self.num_nodes
        return np.array(batch_x), np.array(batch_y)

