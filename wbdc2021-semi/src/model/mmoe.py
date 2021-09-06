# -*- coding:utf-8 -*-
"""
Author:
    Yiyuan Liu, lyy930905@gmail.com
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit
from deepctr.layers.utils import combined_dnn_input, add_func
from deepctr.layers.core import PredictionLayer
from deepctr.layers.activation import activation_layer

from tensorflow.python.keras.initializers import glorot_normal, Zeros, Ones
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2


class MMOELayer(Layer):

    def __init__(self, num_tasks, num_experts, output_dim, l2_reg_dnn, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.l2_reg_dnn = l2_reg_dnn
        self.seed = seed
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            # regularizer=l2(self.l2_reg_dnn),
            initializer=glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                # regularizer=l2(self.l2_reg_dnn),
                initializer=glorot_normal(seed=self.seed+i)))
        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0))
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output)
        return outputs

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


def MMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         use_bn=False,
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: a Keras model instance
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    '''

    sparse_embed = concat_func(sparse_embedding_list, axis=1)
    dense_embed = concat_func(dense_value_list, axis=1)
    
    sparse_embed = tf.transpose(sparse_embed, [0,2,1])
    sparse_embed = tf.keras.layers.Conv1D(1024, kernel_size=3, activation='relu')(sparse_embed)
    sparse_embed = tf.keras.layers.Conv1D(32, kernel_size=1)(sparse_embed)
    sparse_embed = tf.keras.layers.Flatten()(sparse_embed)
    
    dnn_input = tf.keras.layers.Concatenate()([dense_embed, sparse_embed])
    '''
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn=use_bn, seed=seed)(dnn_input)
    # dnn_out = tf.keras.layers.LayerNormalization()(dnn_out)
    # dnn_input = tf.keras.layers.Dense(1024, activation='linear')(dnn_input)
    # dnn_input = tf.keras.layers.BatchNormalization(center=True, scale=True)(dnn_input)
    
    # dnn_out = tf.keras.layers.Dense(dnn_input.shape[-1]*4, activation='relu')(dnn_input)
    # dnn_out = tf.keras.layers.Dense(dnn_input.shape[-1], activation='linear')(dnn_input)
    # dnn_out = dnn_input + dnn_out
    
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        # mmoe_out = tf.keras.layers.Concatenate()([mmoe_out, dnn_input])
        logit = tf.keras.layers.Dense(1, use_bias=True, activation=None, kernel_regularizer=l2(l2_reg_dnn))(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model

#####################################################################################################################################
### Multihead
#####################################################################################################################################

class MultiHeadLayer(Layer):

    def __init__(self, att_embedding_size, head_num, seed=1024, **kwargs):
        self.num_units = att_embedding_size * head_num
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.seed = seed
        super(MultiHeadLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
        super(MultiHeadLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        queries, keys, values = inputs
        querys = tf.tensordot(queries, self.W_Query,
                              axes=(-1, 0))  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(values, self.W_Value, axes=(-1, 0))
        
        # head_num*None T_q D
        querys = tf.concat(tf.split(querys, self.head_num, axis=1), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=1), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=1), axis=0)
        
        # "scaled_dot_product"
        outputs = tf.matmul(querys, keys, transpose_b=True)
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        
        # softmax
        outputs = tf.nn.softmax(outputs, axis=-1)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=1)
        return result

    def get_config(self):

        config = {'att_embedding_size': self.att_embedding_size,
                  'head_num': self.head_num}
        base_config = super(MultiHeadLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (None, self.num_units)
    
class MMOE_Att_Layer(Layer):

    def __init__(self, att_embedding_size, head_num, num_tasks, seed=1024, **kwargs):
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_tasks = num_tasks
        self.units = self.att_embedding_size * self.head_num
        self.seed = seed
        super(MMOE_Att_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(MultiHeadLayer(self.att_embedding_size, self.head_num, seed=self.seed+i))
        super(MMOE_Att_Layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        for i in range(self.num_tasks):
            output = self.gate_kernels[i]([inputs, inputs, inputs])
            outputs.append(output)
        return outputs

    def get_config(self):

        config = {'att_embedding_size': self.att_embedding_size,
                  'head_num': self.head_num,
                  'num_tasks':self.num_tasks}
        base_config = super(MMOE_Att_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.units] * self.num_tasks
    
    
def MMOEAtt(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
            att_embedding_size=128, head_num=8, l2_reg_embedding=1e-5, l2_reg_dnn=0, 
            task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    mmoe_outs = MMOE_Att_Layer(att_embedding_size, head_num, num_tasks)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model


#####################################################################################################################################
### share MMOE  PLE
#####################################################################################################################################

class PLELayer(Layer):

    def __init__(self, num_tasks, num_experts, output_dim, share_dim, l2_reg_dnn, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.share_dim = share_dim
        self.l2_reg_dnn = l2_reg_dnn
        self.seed = seed
        super(PLELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            regularizer=None,
            initializer=glorot_normal(seed=self.seed))
        
        self.share_kernel = self.add_weight(
            name='share_kernel',
            shape=(input_dim, self.share_dim),
            dtype=tf.float32,
            regularizer=None,
            initializer=glorot_normal(seed=self.seed))
        
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                regularizer=None,
                initializer=glorot_normal(seed=self.seed+i)))
        super(PLELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        share_out = tf.tensordot(inputs, self.share_kernel, axes=(-1,0))
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0))
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output+share_out)
        return outputs

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(PLELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


def SMMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    mmoe_outs = PLELayer(num_tasks, num_experts, expert_dim, share_dim=expert_dim, l2_reg_dnn=l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model


#####################################################################################################################################
### Attention
#####################################################################################################################################
    
class MMOEAttLayer(Layer):

    def __init__(self, num_tasks, num_experts, output_dim, l2_reg_dnn, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.l2_reg_dnn = l2_reg_dnn
        self.seed = seed
        super(MMOEAttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            regularizer=l2(self.l2_reg_dnn),
            initializer=glorot_normal(seed=self.seed))
        
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts * self.output_dim),
                dtype=tf.float32,
                regularizer=l2(self.l2_reg_dnn),
                initializer=glorot_normal(seed=self.seed+i)))
        super(MMOEAttLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0))
            gate_out = tf.nn.softmax(gate_out)
            output = expert_out * gate_out
            outputs.append(output)
        return outputs

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOEAttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks
    
    
def MMOEAttention(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
            att_embedding_size=128, head_num=8, l2_reg_embedding=1e-5, l2_reg_dnn=0, 
            task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    mmoe_outs = MMOEAttLayer(att_embedding_size, head_num, num_tasks, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model

#####################################################################################################################################
### Wide & MMoE
#####################################################################################################################################

def MMOEWDL(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
            l2_reg_linear=1e-5,
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())
    
    linear_logit = get_linear_logit(features, dnn_feature_columns, seed=seed, prefix='linear',
                                        l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(add_func([logit, linear_logit]))
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model

#####################################################################################################################################
### GateNet
#####################################################################################################################################
class GateLayer_global(Layer):

    def __init__(self, l2_reg_dnn, seed=1024, **kwargs):
        self.seed = seed
        super(GateLayer_global, self).__init__(**kwargs)
        self.l2_reg_dnn = l2_reg_dnn

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        

        self.gate_kernel = self.add_weight(
                name='gate_embed_weight',
                shape=(input_dim, input_dim),
                dtype=tf.float32,
                regularizer=l2(self.l2_reg_dnn),
                initializer=glorot_normal(seed=self.seed))
        
        super(GateLayer_global, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        gate_attention = tf.tensordot(inputs, self.gate_kernel, axes=(-1, 0)) # 矩阵乘
        gate_attention = tf.tanh(gate_attention)
        outputs = inputs * gate_attention
        return outputs

    def get_config(self):

        config = {}
        base_config = super(GateLayer_global, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
class GateLayer_private(Layer):

    def __init__(self, l2_reg_dnn, seed=1024, **kwargs):
        self.seed = seed
        super(GateLayer_private, self).__init__(**kwargs)
        self.l2_reg_dnn = l2_reg_dnn

    def build(self, input_shape):
        
        self.gate_kernels = []
        for i in range(len(input_shape)):
            self.gate_kernels.append(self.add_weight(
                    name='gate_private_weight_'.format(i),
                    shape=(input_shape[i][-1], input_shape[i][-1]),
                    dtype=tf.float32,
                    regularizer=l2(self.l2_reg_dnn),
                    initializer=glorot_normal(seed=self.seed)))
        
        super(GateLayer_private, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        for i in range(len(inputs)):
            gate_attention = tf.tensordot(inputs[i], self.gate_kernels[i], axes=(-1, 0)) # 矩阵乘
            gate_attention = tf.sigmoid(gate_attention)
            outputs.append(inputs[i] * gate_attention)
        return outputs

    def get_config(self):

        config = {}
        base_config = super(GateLayer_private, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0]] * len(input_shape)


# global 形式
def GateNet(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         use_bn=False,
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    # sparse_embedding_list = GateLayer_private(l2_reg_dnn, seed)(sparse_embedding_list)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    dnn_input = GateLayer_global(l2_reg_dnn, seed)(dnn_input)
    # 首先在dnn input后面加上全局形式的attention

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  use_bn=use_bn, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        mmoe_out = tf.keras.layers.Concatenate()([mmoe_out, dnn_input])
        # mmoe_out = GateLayer_global(l2_reg_dnn, seed)(mmoe_out)
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, kernel_regularizer=l2(l2_reg_dnn))(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model


#####################################################################################################################################
### DNN
#####################################################################################################################################

class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.LayerNormalization(scale=False, center=False) for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
#####################################################################################################################################
### Pretrain
#####################################################################################################################################
from tensorflow.keras import layers
# global 形式
def PretrainModel(feed_classes, user_classes, l2_reg_embedding, l2_reg_dnn, latent_factors):
    
    feed_input = layers.Input(shape=(1,))
    feed_embedding = layers.Embedding(feed_classes, 
                                      latent_factors, 
                                      name='pretrain_feed_embedding', 
                                      embeddings_regularizer=l2(l2_reg_embedding))(feed_input)
    feed_embedding = layers.Flatten()(feed_embedding)

    output = layers.Dense(user_classes, 
                          activation='softmax', 
                          name='user_dense',
                          use_bias=False,
                          kernel_regularizer=l2(l2_reg_dnn))(feed_embedding)

    model = tf.keras.models.Model(inputs=feed_input,
                                  outputs=output)
    return model
    
    
def load_pretrain_model(model, pretrain_model_path, userid_size, feedid_size, embed_size):
    print("Begin load pretrain model")
    pretrain_model = PretrainModel(userid_size+1, feedid_size+1, 0, 0, embed_size)
    pretrain_model.load_weights(pretrain_model_path)
    pretrain_weights = pretrain_model.layers[-1].get_weights()[0].T
        

    for i in range(len(model.layers)):
        if model.layers[i].name=='sparse_emb_feedid':
            #  and model.layers[i].get_weights()[0].shape[0] == vocab_size*embed_size
            model.layers[i].set_weights([pretrain_weights])
            print("End pretrain")
            
    return model


#####################################################################################################################################
### MMOE FM
#####################################################################################################################################
from deepctr.layers.interaction import FM
from itertools import chain
from deepctr.feature_column import DEFAULT_GROUP_NAME
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input

def MMOEFM(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         use_bn=False, fm_group=[DEFAULT_GROUP_NAME],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)
    
    # linear_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  use_bn=use_bn, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])
        mmoe_out = tf.keras.layers.Concatenate()([mmoe_out, dnn_input])

        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, kernel_regularizer=l2(l2_reg_dnn))(mmoe_out)
        logit = add_func([logit, fm_logit])
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model

#####################################################################################################################################
### MMOE FM
#####################################################################################################################################
from deepctr.layers.interaction import FM
from itertools import chain
from deepctr.feature_column import DEFAULT_GROUP_NAME
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input
from deepctr.layers.interaction import FEFMLayer
from deepctr.layers.utils import concat_func, combined_dnn_input, reduce_sum

def MMOEFeFM(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         use_bn=False, fm_group=[DEFAULT_GROUP_NAME],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)

    # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  use_bn=use_bn, seed=seed)(dnn_input)
    
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        # fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])
        
        fefm_interaction_embedding = concat_func([FEFMLayer(
            regularizer=l2_reg_embedding)(concat_func(v, axis=1))
                                                  for k, v in group_embedding_dict.items() if k in [DEFAULT_GROUP_NAME]],
                                                 axis=1)
        fefm_logit = tf.keras.layers.Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=True))(fefm_interaction_embedding)
        
        mmoe_out = tf.keras.layers.Concatenate()([mmoe_out, dnn_input])
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, kernel_regularizer=l2(l2_reg_dnn))(mmoe_out)
        logit = add_func([logit, fefm_logit])
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model


#####################################################################################################################################
### MMOE MVM
#####################################################################################################################################
from tensorflow.python.keras import backend as K
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

class MVM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, l2_reg_dnn, **kwargs):
        self.l2_reg_dnn = l2_reg_dnn
        super(MVM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))
        self.field_num = input_shape[-2]
        # 偏执可以初始化为1
        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            dtype=tf.float32,
            regularizer=l2(self.l2_reg_dnn),
            initializer=glorot_normal(seed=2021))
            
        super(MVM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs
        
        all_order = tf.add(concated_embeds_value, self.bias)
        mvm_func = all_order[:,0,:]
        for i in range(1, self.field_num):
            mvm_func = tf.multiply(mvm_func, all_order[:,i,:])
        cross_term = reduce_sum(mvm_func, axis=1, keep_dims=False)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

def MMOEMVM(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         use_bn=False, fm_group=[DEFAULT_GROUP_NAME],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)
    
    linear_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  use_bn=use_bn, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for i, mmoe_out, task in zip(range(num_tasks), mmoe_outs, tasks):
        fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])
        mvm_logit = add_func([MVM(l2_reg_dnn=l2_reg_dnn)(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])
        linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix=f'linear_{i}', l2_reg=l2_reg_embedding)

        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, kernel_regularizer=l2(l2_reg_dnn))(mmoe_out)
        logit = add_func([logit, fm_logit, linear_logit, mvm_logit])
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model

#####################################################################################################################################
### MMOE AFM
#####################################################################################################################################
from deepctr.layers.interaction import AFMLayer
from itertools import chain
from deepctr.feature_column import DEFAULT_GROUP_NAME
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input

def MMOEAFM(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         use_bn=False, fm_group=[DEFAULT_GROUP_NAME],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)
    
    linear_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  use_bn=use_bn, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for i, mmoe_out, task in zip(range(num_tasks), mmoe_outs, tasks):
        afm_logit = add_func([AFMLayer(l2_reg_w=l2_reg_dnn)(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])
        linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix=f'linear_{i}_', l2_reg=l2_reg_embedding)

        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, kernel_regularizer=l2(l2_reg_dnn))(mmoe_out)
        logit = add_func([logit, afm_logit, linear_logit])
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model

#####################################################################################################################################
### MMOE fwfm
#####################################################################################################################################
from deepctr.layers.interaction import FwFMLayer
from itertools import chain
from deepctr.feature_column import DEFAULT_GROUP_NAME
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input

def MMOEfwfm(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         use_bn=False, fm_group=[DEFAULT_GROUP_NAME],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)
    
    linear_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  use_bn=use_bn, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim, l2_reg_dnn)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for i, mmoe_out, task in zip(range(num_tasks), mmoe_outs, tasks):
        fwfm_logit = add_func([FwFMLayer(regularizer=l2_reg_dnn, num_fields=len(linear_feature_columns))(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])
        linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix=f'linear_{i}_', l2_reg=l2_reg_embedding)

        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, kernel_regularizer=l2(l2_reg_dnn))(mmoe_out)
        logit = add_func([logit, fwfm_logit, linear_logit])
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model