import os
import sys
import networkx as nx
import re
import json
import numpy as np
import math
from queue import Queue

class DataHandler(object):
    @staticmethod
    def load_graph(file_path):
        G = nx.Graph()
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) != 2:
                    continue
                G.add_edge(int(items[0]), int(items[1]))
        return G

    @staticmethod
    def load_tree(file_path):
        G = nx.DiGraph()
        n, m = None, None
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) != 2:
                    continue
                if n is None:
                    n, m = int(items[0]), int(items[1])
                else:
                    G.add_edge(int(items[0]), int(items[1]))
        return G, n, m

    @staticmethod
    def load_fea(file_path):
        X = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                items = line.split()
                if len(items) < 1:
                    continue
                X.append([float(item) for item in items])
        return np.array(X)

    @staticmethod
    def transfer_to_matrix(graph):
        n = graph.number_of_nodes()
        mat = np.zeros([n, n])
        for e in graph.edges():
            mat[e[0]][e[1]] = 1
            mat[e[1]][e[0]] = 1
        return mat

    @staticmethod
    def transfer_to_nx(g_mat):
        G = nx.Graph()
        for i in xrange(len(g_mat)):
            for j in xrange(len(g_mat[i])):
                if g_mat[i][j] == 1:
                    G.add_edge(i, j)
        return G

    @staticmethod
    def normalize_adj_matrix(g):
        # diagonal should be 1
        mat_ret = g / np.sum(g, axis = 1, keepdims = True)
        return mat_ret

    @staticmethod
    def cal_euclidean_distance(x):
        X = np.array(x)
        a = np.square(np.linalg.norm(X, axis = 1, keepdims = True))
        D = -2 * np.dot(X, np.transpose(X)) + a + np.transpose(a)
        return D

    @staticmethod
    def symlink(src, dst):
        try:
            os.symlink(src, dst)
        except OSError:
            os.remove(dst)
            os.symlink(src, dst)


    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as f:
            s = f.read()
            s = re.sub('\s',"", s)
        return json.loads(s)

    @staticmethod
    def append_to_file(file_path, s):
        with open(file_path, "a") as f:
            f.write(s)

    @staticmethod
    def load_ground_truth(file_path):
        lst = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                lst.append([int(i) for i in items])
        lst.sort()
        return [i[1] for i in lst]

