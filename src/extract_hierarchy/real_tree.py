import sys
import os
import networkx as nx

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, '..'))
from utils.env import *
from utils.data_handler import DataHandler as dh
from shared_types import Node

def dfs(u, tree):
    if len(tree[u].childst) == 0:
        tree[u].coverst = set([u])
        return
    for v in tree[u].childst:
        dfs(v, tree)
        tree[u].coverst = tree[u].coverst | tree[v].coverst

def extract_hierarchy(G, params):
    g, n, m = dh.load_tree(os.path.join(DATA_PATH, params["file_path"]))
    tree = [None] * n
    for u in g:
        tree[u] = Node(u, set(g[u].keys()), set())
    dfs(n - 1, tree)
    return tree
