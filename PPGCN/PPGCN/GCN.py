#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf

_LAYER_UIDS = {}

# 初始化shape大小的全0矩阵（tf.Variable）
def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# 点乘【非矩阵乘法！】
# 一般在卷积层，都会使用多个过滤器来采集图片的特征
def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    # 三个参数:inputs, filters(输出空间的维度，即卷积过滤器的数量）, kernel_size
    #   input：输入图像，[图片数量, 图片高度, 图片宽度, 图像通道数]，一个4维的tensor
    #   filter：卷积核大小，[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，通道数应该与input的通道数相同
    res = tf.layers.conv2d(x, y[1], [1, y[0]])  # 卷积层的函数接口 这个层创建了一个卷积核，将输入进行卷积来输出一个 tensor
    return res[:, :, 0, :]  # 第3维只要第0个数字    【x_shape0,x_shape1, x_shape2-y[0]+1, y[1]】


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            # assert后面的字符串是异常参数，用来解释断言并更好的知道是哪里出了问题
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()  # 获取小写的类名
            name = layer + '_' + str(get_layer_uid(layer))  # layer_1,layer_2,layer_3 .... etc
        self.name = name
        self.vars = {}


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input, adj_matrix, output_dim, dropout=0., act=tf.nn.relu, bias=False,
                 **kwargs):
        # 对继承自父类的属性进行初始化
        # 首先找到test的父类（比如是类A），然后把类test的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数
        super(GraphConvolution, self).__init__(**kwargs)    # 可能指定name，也可能不指定

        self.dropout = dropout  # 避免过拟合？使矩阵一部分变为0，另一部分则变为element/keep_prob

        self.act = act  # 激活函数
        self.adj_matrix = adj_matrix    # 相似度矩阵

        self.bias = bias    # 偏差项/偏置量，类型？？？
        self.input = input  # xdata??? 节点的属性
        self.output_dim = output_dim    # an integer

        with tf.variable_scope(self.name + '_vars'):
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def call(self):
        with tf.name_scope(self.name):
            outputs = self._call(self.input)
        return outputs

    def _call(self, inputs):
        # print(inputs)
        x = inputs
        x = tf.nn.dropout(x, self.dropout)  # 激活函数，根据概率keep_prob(第二个参数)独立决定每个神经元是否被抑制

        # convolve / 卷
        x = tf.matmul(self.adj_matrix, x)   # matmul([1,5000,5000],[1,5000,1000/512/256])==>[1,5000,1000/512/256]
        print(x.shape)
        # tf.expand_dims 为Tensor的shape在指定维度的索引轴处 增加一个为1的维度。
        # 若指定轴的负数，则从最后向后计数
        pre_sup = dot(tf.expand_dims(x, -1), [int(self.input.shape[-1]), int(self.output_dim)])
        output = pre_sup

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output) # shape：[input_shape0, input_shape1, output_dim[-1]


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

    def build(self):
        raise NotImplementedError


class GCN(Model):
    def __init__(self, x, adj_matrix, output_dim, dropout=0.2, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input = x
        self.adj_matrix = adj_matrix
        self.dropout = dropout
        self.output_dim = output_dim

    def build(self):
        # 若没有指定输出维度[]
        if len(self.output_dim) == 0:
            return self.input

        outputs = GraphConvolution(input=self.input,
                                   adj_matrix=self.adj_matrix,
                                   output_dim=self.output_dim[0],
                                   act=tf.nn.relu,
                                   dropout=self.dropout).call()

        # 多次训练
        for i in range(1, len(self.output_dim)):
            outputs = GraphConvolution(input=outputs,
                                       adj_matrix=self.adj_matrix,
                                       output_dim=self.output_dim[i],
                                       act=tf.nn.relu,
                                       dropout=self.dropout).call()

        return outputs