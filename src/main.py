#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import re
import json
import math
import argparse
import time
import subprocess
import numpy as np
import networkx as nx
import tensorflow as tf
import datetime
from operator import itemgetter

from get_network_hierarchy.get_network import GetNetwork as gn
from utils.batch_strategy import BatchStrategy
from utils.env import *
from utils.data_handler import DataHandler as dh
from utils.metric import Metric

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def dfs(u, tree, handlers, params, res_radius, res_coordinates):
    if len(tree[u].childst) == 0:
        return

    node_in_tree, sim_mat, var_mat = handlers["get_network"].get_network(u, tree)


    if (len(node_in_tree) <= 2):
        rc = np.random.random(size = params["embedding_model"]["embedding_size"]) * 2 - 1
        rc_b = rc / np.linalg.norm(rc) * res_radius[u]
        rc = rc_b + res_coordinates[u]
        res_coordinates[node_in_tree[0]] = rc
        if (len(node_in_tree) == 2):
            rc_b = - rc_b + res_coordinates[u]
            res_coordinates[node_in_tree[1]] = rc_b
    else:
        # neural embedding
        sim_mat_norm = dh.normalize_adj_matrix(sim_mat)
        if "batch_params" in params:
            bs = BatchStrategy(sim_mat_norm, params["batch_params"])
        else:
            bs = BatchStrategy(sim_mat_norm, {})
        params["embedding_model"]["num_nodes"] = len(sim_mat_norm)
        ne = handlers["embedding_model"](params["embedding_model"])
        X = ne.train(getattr(bs, params["embedding_model"]["batch_func"]), params["embedding_model"]["iteration"])

        del ne, bs

        # transfer embedding
        params["transfer_embeddings"]["num_nodes"] = len(sim_mat_norm)

        te = handlers["transfer_embeddings"](params["transfer_embeddings"])
        Z, dic = te.transfer(X, res_coordinates[u], res_radius[u], params["transfer_embeddings"]["iteration"])
        for i in xrange(len(node_in_tree)):
            res_coordinates[node_in_tree[i]] = Z[i]

        del te, dic, sim_mat_norm

    # cal radius
    r = np.zeros(len(node_in_tree), dtype = np.float32)
    cnt = np.zeros(len(r), dtype = np.float32)
    for i in xrange(len(r)):
        for j in xrange(len(r)):
            if sim_mat[i][j] > sys.float_info.epsilon:
                r[i] += var_mat[i][j] / (sim_mat[i][j] * params["scaling_radius"]) * np.linalg.norm(res_coordinates[node_in_tree[i]] - res_coordinates[node_in_tree[j]])
                cnt[i] += 1.0

    for i in xrange(len(r)):
        if cnt[i] > sys.float_info.epsilon:
            r[i] = r[i] / cnt[i]
        res_radius[node_in_tree[i]] = min(params["radius_max"] * res_radius[u], max(params["radius_min"] * res_radius[u], r[i]))

    del r, cnt, node_in_tree, sim_mat, var_mat


    for v in tree[u].childst:
        dfs(v, tree, handlers, params, res_radius, res_coordinates)


def train_model(params):
    g_mat, tree = extract_tree(params)

    handlers = {}
    handlers["get_network"] = gn(g_mat, params["get_network_hierarchy"])
    handlers["embedding_model"] = __import__('node_embedding.' + params["embedding_model"]["func"], fromlist = ["node_embedding"]).NodeEmbedding
    handlers["transfer_embeddings"] = __import__('transfer_embeddings.' + params["transfer_embeddings"]["func"], fromlist = ["transfer_embeddings"]).TransferEmbedding

    res_coordinates = [None] * len(tree)
    res_coordinates[len(tree) - 1] = np.zeros(params["embedding_model"]["embedding_size"], dtype = np.float32)
    res_radius = [None] * len(tree)
    res_radius[len(tree) - 1] = float(params["init_radius"])
    dfs(len(tree) - 1, tree, handlers, params, res_radius, res_coordinates)

    res_path = params["train_output"]
    dh.symlink(res_path, os.path.join(RES_PATH, "new_train_res"))
    dh.append_to_file(res_path, json.dumps({"radius": np.array(res_radius).tolist(), "coordinates": np.array(res_coordinates).tolist()}))

    return res_coordinates, res_radius

def extract_tree(params):
    g = dh.load_graph(os.path.join(DATA_PATH, params["network_file"]))
    g_mat = dh.transfer_to_matrix(g)
    eh = __import__('extract_hierarchy.' + params["extract_hierarchy_model"]["func"], fromlist = ["extract_hierarchy"])
    tree = eh.extract_hierarchy(g, params["extract_hierarchy_model"], )

    return g_mat, tree


def metric(params):
    js = json.loads(open(params["metric_input"]).read())
    coordinates = np.array(js["coordinates"])
    radius = np.array(js["radius"])
    res_path = params["metric_output"]
    dh.symlink(res_path, os.path.join(RES_PATH, "new_metric_res"))
    ret = []
    for metric in params["metric_function"]:
        if metric["metric_func"] == "draw_circle_2D":
            pic_path = os.path.join(PIC_PATH, "draw_circle_" + str(int(time.time() * 1000.0)) + ".pdf")
            dh.symlink(pic_path, os.path.join(PIC_PATH, "new_draw_circle"))
            getattr(Metric, metric["metric_func"])(coordinates, radius, metric, params["num_nodes"], pic_path)
        else:
            origin_coordinates = coordinates[: params["num_nodes"]]
            res = getattr(Metric, metric["metric_func"])(origin_coordinates, metric)
            ret.append((metric["metric_func"], res))
    dh.append_to_file(res_path, json.dumps(ret))

    return ret


def main():

    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--operation', type = str, default = "all", help = "[all | extract_tree | train | metric]")
    parser.add_argument('--conf', type = str, default = "default")
    parser.add_argument('--metric_input', type = str, default = "new_train_res")
    parser.add_argument('--train_output', type = str, default = str(int(time.time() * 1000.0)))
    parser.add_argument('--metric_output', type = str, default = str(int(time.time() * 1000.0)))

    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))
    params["metric_input"] = os.path.join(RES_PATH, args.metric_input)
    params["train_output"] = os.path.join(RES_PATH, "train_res_" + args.train_output)
    params["metric_output"] = os.path.join(RES_PATH, "metric_res_" + args.metric_output)


    if args.operation == "all":
        train_model(params)
        metric(params)
    elif args.operation == "extract_tree":
        extract_tree(params)
    elif args.operation == "train":
        train_model(params)
    elif args.operation == "metric":
        metric(params)
    else:
        print "Not Support!"

if __name__ == "__main__":
    main()
