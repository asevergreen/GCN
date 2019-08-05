#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import csv
from GCN import *
from Constants import *

data_size =  DATA_SIZE # same_size
node_size = NODE_SIZE
node_embedding = 1000  # 映射成1000维的特征向量,初始设定有1000个event instance

# GCN
gcn_para = [512, 256]

# train
batch_size = 64
# epoch_num = 150
epoch_num = 200
learning_rate = 1e-3  # 0.001
momentum = 0.9  # 加速收敛,动量


def read_data():
    diff = np.load(INTERMEDIATE_DATA_DIR+"diff.npy")
    same = np.load(INTERMEDIATE_DATA_DIR+"same.npy")
    index = [i for i in range(data_size)]
    np.random.shuffle(index)
    diff = diff[index]  # 乱序读入
    np.random.shuffle(index)
    same = same[index]
    # PairWise
    diff_label = np.zeros((int(data_size),))
    same_label = np.ones((int(data_size),))
    # 分成两部分，训练集和测试集
    boundary = (int)(0.7*data_size)
    return diff[0:boundary], same[0:boundary], diff_label[0:boundary], same_label[0:boundary],\
           diff[boundary:], same[boundary:], diff_label[boundary:], same_label[boundary:]


diff, same, diff_label, same_label, test_diff, test_same, test_difflabel, test_samelabel = read_data()


def get_test_data():
    r = np.concatenate((test_diff, test_same), axis=0)  # 将test_diff和test_same拼接
    l = np.concatenate((test_difflabel, test_samelabel))
    index = [i for i in range(len(r))]
    np.random.shuffle(index)
    # 用同一个随机顺序打乱
    r = r[index]
    l = l[index]
    return r, l


def get_data(ix, int_batch):
    if ix + int_batch >= data_size:
        ix = data_size - int_batch
        end = data_size
    else:
        end = ix + int_batch
    # d = diff[index:end,:]
    # s = same[index:end,:]
    r = np.concatenate((diff, same), axis=0)
    l = np.concatenate((diff_label, same_label))
    index = [i for i in range(len(r))]
    np.random.shuffle(index)
    r = r[index]
    l = l[index]
    return r[ix:end], l[ix:end]


# In[8]:

class PPGCN(object):
    def __init__(self, session, # tf会话
                 meta,  # meta-path???
                 nodes, # 节点？？？
                 class_size,    # ？？？
                 gcn_output1,
                 gcn_output2,
                 embedding,
                 batch_size):
        self.meta = meta
        self.nodes = nodes
        self.class_size = class_size
        self.gcn_output1 = gcn_output1
        self.gcn_output2 = gcn_output2
        self.embedding = embedding
        self.batch_size = batch_size

        self.build_placeholders()

        self.loss, self.probabilities, self.weight, self.v1, self.v2 = self.forward_propagation()
        # self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), tf.trainable_variables())
        self.pred = tf.to_int32(2 * self.probabilities) # 从0~1区间的小数转换为0或1
        correct_prediction = tf.equal(self.pred, self.t)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('Forward propagation finished.')

        self.sess = session
        self.optimizer = tf.train.MomentumOptimizer(self.lr, self.mom).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables())
        print('Backward propagation finished.')

    def build_placeholders(self):
        self.a = tf.placeholder(tf.float32, [self.meta, self.nodes, self.nodes], 'adj')
        self.x = tf.placeholder(tf.float32, [self.nodes, self.embedding], 'nxf')    # input，doc2vec
        # self.t = tf.placeholder(tf.int32, [None], 'labels')    # shape默认是None，就是一维值，也可以是多维
        # self.p1 = tf.placeholder(tf.int32, [None], 'left_pair')  # pair-wise
        # self.p2 = tf.placeholder(tf.int32, [None], 'right_pair')
        self.t = tf.placeholder(tf.int32, [None], 'labels')    # shape默认是None，就是一维值，也可以是多维
        self.p1 = tf.placeholder(tf.int32, [None], 'left_pair')  # pair-wise
        self.p2 = tf.placeholder(tf.int32, [None], 'right_pair')
        print(self.p1)
        print(self.p2)
        self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.placeholder(tf.float32, [], 'momentum')

    def forward_propagation(self):
        # node_size = 5000:
        # input ==> (13, 5000, 5000)
        # self.x ==> (5000, 1000)
        # self.t ==> (5000,)
        # attention ==> (1, 5000, 2084)
        # init 权重矩阵
        with tf.variable_scope('weights_n'):
            A = tf.reshape(self.a, [self.meta, self.nodes * self.nodes])
            A_ = tf.transpose(A, [1, 0])  # 转置，第二个参数是新的维度序列
            # 返回一个用于初始化权重的初始化程序 “Xavier” 。这个初始化器是用来保持每一层的梯度大小都差不多相同。
            W = tf.nn.sigmoid(
                tf.get_variable('W', shape=[self.meta, 1], initializer=tf.contrib.layers.xavier_initializer()))
            weighted_adj = tf.matmul(A_, W)
            # (5000*5000, 1)
            weighted_adj = tf.reshape(weighted_adj, [1, self.nodes, self.nodes])

        with tf.variable_scope('spectral_gcn'):
            # gcn_out: [1, 5000, class_size]
            gcn_out = GCN(tf.expand_dims(self.x, 0), weighted_adj,  # x是个2维tensor
                          [self.gcn_output1, self.gcn_output2, self.class_size]).build()    # 【512，256，128】
            print("gcn_out shape:",gcn_out.shape)

        '''        
        with tf.variable_scope('attention_n'):
            attention_output = attention_model.inference(gcn_out,64,0,hid_units,n_heads,nonlinearity,residual)
        '''
        with tf.variable_scope('extract_n'):
            p1 = tf.one_hot(self.p1, tf.cast(self.nodes,tf.int32))  # 第一个参数指定独热位置,第2个参数是每个独热向量的维度
            p1 = tf.matmul(p1, gcn_out[0])
            p2 = tf.one_hot(self.p2, tf.cast(self.nodes,tf.int32))
            p2 = tf.matmul(p2, gcn_out[0])  # gcn_out[0]: [5000,class_size]


        with tf.variable_scope('cosine'):
            # p ==> (batch_size, feature)
            p1_norm = tf.sqrt(tf.reduce_sum(tf.square(p1), axis=1)) # tf.reduce_sum: arg0表示要求和的数据，arg1=0表示纵向求和，=1横向求和
            p2_norm = tf.sqrt(tf.reduce_sum(tf.square(p2), axis=1))
            #            p1_p2 = tf.reduce_sum(tf.multiply(p1, p2), axis=1)
            #            cosine = p1_p2 / (p1_norm * p2_norm)
            #            c = tf.expand_dims(cosine, -1)
            #            prob = tf.concat([-c, c], axis=1)
            c = p1_norm / p2_norm
            c = tf.expand_dims(c, -1)  # batch, 1
            c = tf.concat([c, 1 / c], axis=1)   # 将多个张量在维度上合并
            true_prob = -tf.math.log(tf.reduce_max(c, axis=1) - 1 + 1e-8)    # 同类得出的结果区间应在0~8
            true_p = tf.expand_dims(true_prob, -1)
            prob = tf.concat([-true_p, true_p], axis=1)

            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.one_hot(self.t, 2), logits=prob)   # logits: 神经网络最后一层的输出
            # loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.one_hot(self.t, 2), logits=prob)   # logits: 神经网络最后一层的输出

        # tf.nn.sigmoid将true_prob映射到（0，1）区间
        return loss, tf.nn.sigmoid(true_prob), W, p1[0], p2[0]

    def train(self, x, a, t, p1, p2, learning_rate=1e-2, momentum=0.9):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.t: t,
            self.p1: p1,
            self.p2: p2,
            self.lr: learning_rate,
            self.mom: momentum
        }
        _, loss, acc, pred, w, v1, v2, prob = self.sess.run(
            [self.optimizer, self.loss, self.accuracy, self.pred, self.weight, self.v1, self.v2, self.probabilities],
            feed_dict=feed_dict)

        return loss, acc, pred, w, v1, v2, prob[0]

    def test(self, x, a, t, p1, p2):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.t: t,
            self.p1: p1,
            self.p2: p2
            #   self.lr: learning_rate,
            #   self.mom: momentum
        }
        acc, pred = self.sess.run([self.accuracy, self.pred], feed_dict=feed_dict)
        return acc, pred

    def pred(self, x, a, t, p1, p2):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.p1: p1,
            self.p2: p2
        }
        pred = self.sess.run([self.pred], feed_dict = feed_dict)
        return pred


# 计算f1值判断模型准确性
def com_f1(pred, label):
    MI_F1 = []
    l = len(pred)
    TP = 0  # 预测为正向（P），实际上预测正确（T），即判断为正向的正确率
    FP = 0  # 预测为正向（P），实际上预测错误（F），误报率，即把负向判断成了正向
    FN = 0  # 预测为负向（N），实际上预测错误（F），漏报率，即把正向判断称了负向
    TN = 0  # 预测为负向（N），实际上预测正确（T），即判断为负向的正确率
    f1 = 0
    for i in range(l):
        if pred[i] == 1 and label[i] == 1:
            TP += 1
        elif pred[i] == 1:
            FP += 1
        elif label[i] == 1:
            FN += 1
        else:
            TN += 1
    if TP + FP == 0:
        pre = 0
    else:
        pre = TP / (TP + FP)  # 精确率Precision=TP / （TP+FP），即在预测为正向的数据中，有多少预测正确了
    if TP + FN == 0:
        rec = 0
    else:
        rec = TP / (TP + FN)  # 召回率Recall=TP / （TP+FN），即在所有正向的数据中，有多少预测正确了
    acc = (TP + TN) / l  # 准确率Accuracy=（TP+TN） / （TP+FP+TN+FN）， 即预测正确的比上全部的数据
    if (pre + rec) != 0:
        f1 = 2 * pre * rec / (pre + rec)
    return [pre, rec, acc, f1]


if __name__ == "__main__":
    xdata = np.load(INTERMEDIATE_DATA_DIR+"xdata.npy")
    adj_data = np.load(INTERMEDIATE_DATA_DIR+"adj_data.npy")
    test_data, test_label = get_test_data()
    PRAF = []

    with tf.Session() as sess:
        net = PPGCN(class_size=128, gcn_output1=gcn_para[0],
                    gcn_output2=gcn_para[1], meta=1, nodes=node_size,
                    session=sess, embedding=node_embedding, batch_size=batch_size)
        sess.run(tf.global_variables_initializer())

        minloss = maxacc = 0
        max_acc = 0
        t = int(batch_size)
        y_=[]
        for epoch in range(epoch_num):

            train_loss = 0
            train_acc = 0
            count = 0
            # 包括了训练集和测试集
            for index in range(0, data_size, t):
                pdata, label = get_data(index, t)
                loss, acc, pred, w, v1, v2, prob = net.train(xdata, adj_data, label, pdata[:, 0], pdata[:, 1],
                                                             learning_rate, momentum)
                y_.append(acc)

                if index % 1 == 0:
                    print("loss: {:.4f} ,acc: {:.4f}".format(loss, acc))
                #                    print(pdata[0])
                #                    print(v1)
                #                    print(v2)
                #                    print(prob)
                if index % 320 == 0:  # 每五轮test一次？？
                    print("NOW:______________epoch:%d\tindex:%d______________"%(epoch, index))
                    # print(test_data[:,0].shape)
                    # print(test_data[:,1].shape)
                    # 正确率
                    eva_acc, eva_pred = net.test(xdata, adj_data, test_label, test_data[:, 0], test_data[:, 1])
                    PRAF.append(com_f1(eva_pred, test_label))
                    # with open("test_acc.txt", "a+") as f:
                    #     f.write(str(eva_acc))
                    #     f.write("\n")
                    print('------------------------------------------------------Test acc: {:.4f}'.format(eva_acc))
                if acc > max_acc:
                    max_acc = acc
                    # f = open('results_new.txt', 'w')
                    # f.write('batch accuracy:' + str(acc))
                    # f.write('\n')
                    # f.write('weight:' + str(w))
                    # f.write('\n')
                    # f.close()
                    print('batch accuracy:', acc)
                    print('golden label:', label)
                    print('pred label:', pred)
                    print('weight:', w)
                    net.saver.save(sess, MODEL_DIR + "model")
                    print('********************* Model Saved *********************')
                train_loss += loss

                train_acc += acc

                count += 1
            train_loss = train_loss / count
            train_acc = train_acc / count
            # with open("train_acc.txt", "a+") as f:
            #     f.write(str(train_acc))
            #     f.write("\n")
            print("epoch{:d} : , train_loss: {:.4f} ,train_acc: {:.4f}".format(epoch, train_loss, train_acc))
            with open("PRAF.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Pre", "Rec", "Acc", "F1"])
                for d in PRAF:
                    writer.writerow(d)
            # print("PRAF:",PRAF)
            if maxacc < train_acc:
                maxacc = train_acc
                minloss = train_loss
        print("train end!")
        print("The loss is {:.4f},The acc is{:.4f}".format(minloss, maxacc))
        x_ = np.linspace(1,len(y_),len(y_))
        plt.plot(x_,np.array(y_,dtype=float))
        plt.show()