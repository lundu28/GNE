import numpy as np
import tensorflow as tf

class TransferEmbedding(object):
    def __init__(self, params):
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.embedding_size = params["embedding_size"]
        self.num_nodes = params["num_nodes"]
        self.lbd = params["lambda"]
        self.theta = params["theta"]
        self.clip_min = params["clip_min"]
        self.tol = params["tol"] if "tol" in params else 0.0001

        def clip_by_min(x, m = 0.0):
            return tf.clip_by_value(x, m, float('inf'))

        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            self.D = tf.placeholder(tf.float32, shape = [self.num_nodes, self.num_nodes])

            self.Z = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0), name = "Z", dtype = tf.float32)

            # shape(a) = [n, 1]
            self.a = tf.norm(self.Z, axis = 1, keep_dims = True)
            self.dist = 2 - 2 * tf.matmul(self.Z, tf.transpose(self.Z)) / clip_by_min(self.a * tf.transpose(self.a), self.clip_min)
            self.D_norm = tf.realdiv(self.D, tf.norm(self.D))
            self.loss = tf.norm(clip_by_min(self.D_norm - tf.realdiv(self.dist, clip_by_min(tf.norm(self.dist), self.clip_min)))) + self.lbd * tf.exp(-self.theta * tf.norm(tf.realdiv(self.dist, clip_by_min(tf.norm(self.dist), self.clip_min))))

            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)


    def train(self, D, epoch_num = 10001, save_path = None):
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            pre = float('inf')
            for i in xrange(epoch_num):
                self.train_step.run({self.D : D})
                if i % 100 == 0:
                    loss = self.loss.eval({self.D : D})
                    if (i % 1000 == 0):
                        print(loss)
                    if abs(loss - pre) < self.tol:
                        break
                    else:
                        pre = loss
            if save_path is not None:
                saver = tf.train.Saver()
                saver.save(sess, save_path)
            z = sess.run(self.Z)
            a = np.linalg.norm(z, axis = 1, keepdims = True)
            return z / a, sess.run(self.dist)

    def load_model(self, save_path):
        with tf.Session(graph = self.tensor_graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            z = sess.run(self.Z)
            return z / np.linalg.norm(z, axis = 1, keepdims = True), sess.run(self.dist)

    def transfer(self, X, xc, r, epoch_num = 10001, save_path= None):
        X = np.array(X)
        xc = np.array(xc)
        a = np.square(np.linalg.norm(X, axis = 1, keepdims = True))
        D = -2 * np.dot(X, np.transpose(X)) + a + np.transpose(a)
        Z, dic = self.train(D, epoch_num, save_path)
        Z = Z * r + xc
        return Z, dic


def main():
    params = {'learn_rate': 0.001, 'embedding_size': 2, 'num_nodes': 3, 'clip_min' : 1e-7, 'theta' : 3, 'lambda' : 0.2, 'optimizer': 'AdamOptimizer'}
    cli = TransferEmbedding(params)
    X = np.array([[0,0], [3, 0], [0, 4]], dtype = np.float32)
    xc = [1, 1]
    r = [3]
    Z = cli.transfer(X, xc, r)
    print Z

if __name__ == "__main__":
    main()
