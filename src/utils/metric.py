import os
import sys
import time
import networkx as nx
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib import colors
from matplotlib.patches import Ellipse, Circle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


from env import *
from data_handler import DataHandler as dh
from lib_ml import MachineLearningLib as mll

class Metric(object):
    @staticmethod
    def cal_metric(TP, FP, TN, FN):
        res = {}
        res["acc"] = float(TP + TN) / float(TP + FP + FN + TN)
        res["precision"] = float(TP) / float(TP + FP) if TP + FP > 0 else 1.0
        res["recall"] = float(TP) / float(TP + FN) if TP + FN > 0 else 1.0
        try:
            res["F1"] = 1.0 / (1.0 / res["recall"] + 1.0 / res["precision"])
        except ZeroDivisionError:
            res["F1"] = 0.0
        return res

    @staticmethod
    def draw_pr(precision, recall, file_name = "pr.png"):
        index = np.array(range(len(precision)))
        width = 0.3
        tmplist1 = [(x, precision[x]) for x in precision]
        tmplist2 = [(x, recall[x]) for x in recall]
        tmplist1.sort()
        tmplist2.sort()
        X = [x[0] for x in tmplist1]
        y1 = [x[1] for x in Gtmplist1]
        y2 = [x[1] for x in tmplist2]
        plt.bar(index - width / 2, y2, width, color = "blue", label="recall")
        plt.bar(index + width / 2, y1, width, color = "red", label="precision")
        plt.grid(True, which='major')
        plt.grid(True, which='minor')
        plt.xticks(index, X, rotation = 45, size = 'small')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=5)

        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def draw_circle_2D(c, r, params, num_nodes = 0, file_path = 'circle.pdf'):
        c_map = params["color_list"]
        x = np.array(c)
        n = len(x) # nx2

        c_id = np.random.randint(0, len(c_map) - 1, size = n)
        cValue=[c_map[id] for id in c_id]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #draw circle
        for i in xrange(num_nodes, n):
            ax.add_patch(Circle(xy=(x[i][0],x[i][1]), radius=r[i], fill=False, ec = cValue[i], alpha=1))
        # draw scatter
        ax.scatter(x[:, 0], x[:, 1], c = cValue, marker='x')
        plt.axis('scaled')

        pp = PdfPages(file_path)
        pp.savefig(fig)
        pp.close()

    @staticmethod
    def cal_euclidean_fractal(nodes, params):
        d = nodes.shape[1]
        xmin = np.amin(nodes, axis=0)
        xmax = np.amax(nodes, axis=0)
        count = np.size(nodes, 0)
        L = max(xmax-xmin)

        nsample = params["nsample"]
        max_tries = params["max_tries"]

        l_list=[]

        N_list = []
        l = L * 2.0
        N = 1

        def rn_box_cover(l):
            N_array = np.zeros(nsample)
            for i in xrange(nsample):
                cover = dict()
                x0 = xmin - np.random.uniform(0, l, size=d)
                #print(x0)
                for j in xrange(count):
                    foo = (nodes[j] - x0) / l
                    cover[tuple(foo.astype(int))]=1
                    #cover[ np.floor((nodes[j]-x0))/l ] = 1
                N_array[i] = len(cover)
            # print(cover)
            return min(N_array)

        def _unique(l_l, N_l):
            if len(N_l) <= 1:
                return (l_l, N_l)

            N_last = N_l[-1]

            same_count = 0

            for i in range(2, len(N_l)):
                if(N_l[-i] == N_last):
                    same_count += 1
            if same_count>0:
                return (l_l[:-same_count], N_l[:-same_count])
            else:
                return (l_l, N_l)


        for k in range(max_tries):
            N = rn_box_cover(l)
            l_list.append(l)
            N_list.append(N)
            l = l / 2

        (l_list, N_list) = _unique(l_list, N_list)
        log_l = np.array([np.log2(item) for item in l_list])
        log_N = np.array([np.log2(item) for item in N_list])

        f = np.poly1d(np.polyfit(-log_l, log_N, 1))

        plt.plot(-log_l, log_N)
        #plt.savefig("save_path")
        #print(f)
        return f(0)

    @staticmethod
    def classification(X, params):
        X_scaled = scale(X)
        y = dh.load_ground_truth(os.path.join(DATA_PATH, params["ground_truth"]))
        acc = 0.0
        for _ in xrange(params["times"]):
             X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = params["test_size"], stratify = y)
             clf = getattr(mll, params["classification_func"])(X_train, y_train)
             acc += mll.infer(clf, X_test, y_test)[1]
        acc /= float(params["times"])
        return acc

def test_cal_euclidean_fractal():
    nodes = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    params = {"nsample": 3000, "max_tries": 30}
    print(Metric.cal_euclidean_fractal(nodes, 3, params))

def main():
    test_cal_euclidean_fractal()

if __name__ == "__main__":
    main()
