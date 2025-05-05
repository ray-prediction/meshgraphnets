# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model for RayThings."""

import sonnet as snt
import tensorflow.compat.v1 as tf

from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization

from absl import logging


max_feat = 12

class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, name='Model'):
    super(Model, self).__init__(name=name)
    # TODO: 
    with self._enter_variable_scope():
      self.random = 0
      self._learned_model = learned_model
      self._output_normalizer = normalization.Normalizer(
          size=1, name='output_normalizer') # TODO: size 1?
      self._node_normalizer = normalization.Normalizer(
          size=max_feat+common.RayNodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = normalization.Normalizer(
          size=4, name='edge_normalizer')  # 2D coord + 3D coord + 2*length = 7 # TODO: 3D 
      
  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes

    # take a single tx location (randomly sampled) and embed a single feature 100 or something idk
    tx_pos = inputs['tx_loc']
    self.random = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(tx_pos)[0], dtype=tf.int32)
    tx_pos = tf.gather(tx_pos, self.random) # shape (,3)

    rx_pos = inputs['rx_loc'] # shape (num_rx, 3)
    # rx locations should be all, embed a 0?

    prim_vertices = inputs['prim_vertices']
    prim_averages = tf.math.reduce_sum(prim_vertices, axis=1, keepdims=True) / 3
    prim_vertices = prim_vertices - prim_averages

    prim_normals = tf.expand_dims(inputs['prim_normals'], axis=-1)

    # graph nodes should be 3d vertices - their average wit normal info included normal
    primitive_features = tf.concat([prim_vertices, prim_normals], -1)
    primitive_features = tf.reshape(primitive_features, [-1, max_feat])
    tx_features = tf.ones([1, 1])
    rx_features = tf.zeros([tf.shape(rx_pos)[0], 1])

    tx_features = tf.pad(tx_features, [[0, 0], [0, max_feat - 1]])
    rx_features = tf.pad(rx_features, [[0, 0], [0, max_feat - 1]])

    primitive_type = tf.fill([tf.shape(primitive_features)[0]], common.RayNodeType.PRIMITIVE)
    tx_type = tf.fill([1], common.RayNodeType.TX)
    rx_type = tf.fill([tf.shape(rx_pos)[0]], common.RayNodeType.RX)

    size = common.RayNodeType.SIZE
    primitive_type_oh = tf.one_hot(primitive_type, size)
    tx_type_oh = tf.one_hot(tx_type, size)
    rx_type_oh = tf.one_hot(rx_type, size)

    primitive_embed = tf.concat([primitive_features, primitive_type_oh], axis=-1)
    tx_embed = tf.concat([tx_features, tx_type_oh], axis=-1)
    rx_embed = tf.concat([rx_features, rx_type_oh], axis=-1)  

    # shape (?, 15) with rows of prims, tx, then rx
    node_features = tf.concat([primitive_embed, tx_embed, rx_embed], axis=0)
    print('node_feat_shape',node_features.shape)


    # make edges next
    graph_adj = inputs['scene_adj']
    tx_adj = inputs['tx_adj'][self.random] # only 1 tx for now
    rx_adj = inputs['rx_adj']

    # print(graph_adj)
    # print(tx_adj)
    # print(rx_adj)

    senders, receivers = common.adj_lists_to_edges(graph_adj, tx_adj, rx_adj)
    # print(senders)
    # print(receivers)

    tx_pos = tf.expand_dims(tx_pos, axis=0)
    prim_pos = tf.reshape(prim_averages, [-1, 3])
    # print(prim_pos.shape)
    points = tf.concat([prim_pos, tx_pos, rx_pos], axis=0)
    # print(points.shape)

    relative_pos = (tf.gather(points, senders) - tf.gather(points, receivers))

    edge_features = tf.concat([relative_pos, tf.norm(relative_pos, axis=-1, keepdims=True)], axis=-1)
    # print(edge_features.shape)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)

    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

  def _build(self, inputs):
    print('called _build')
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  @snt.reuse_variables
  def loss(self, inputs):
    """L2 loss on position."""
    graph = self._build_graph(inputs, is_training=True)
    network_output = self._learned_model(graph)

    # build target total power received
    real_coeff = inputs['real_channel_coeff'][:, self.random]
    imag_coeff = inputs['imag_channel_coeff'][:, self.random]
    # print(imag_coeff.shape)

    target_inco_sum_power = tf.math.reduce_sum(real_coeff ** 2 + imag_coeff ** 2, axis=1)
    # print(incoherent_sum_power)

    # print(tf.math.count_nonzero(incoherent_sum_power))
    num_prim = tf.shape(inputs['prim_vertices'])[0]
    num_tx = 1
    num_rx = tf.shape(inputs['rx_loc'])[0]
    total = num_prim + num_tx + num_rx

    loss_mask = tf.range(total)
    # print(num_prim)
    loss_mask = loss_mask >= (total - num_rx)
    # print(loss_mask[0:56])

    # print(target_inco_sum_power.shape)
    # print(network_output[loss_mask].shape)
    error = tf.reduce_sum((target_inco_sum_power - network_output[loss_mask])**2, axis=1)
    # print(error.shape)
    loss = tf.reduce_mean(error)
    return loss

  # def _update(self, inputs, per_node_network_output):
  #   """Integrate model outputs."""
  #   return -1
  #   print('called _update')
  #   acceleration = self._output_normalizer.inverse(per_node_network_output)
  #   # integrate forward
  #   cur_position = inputs['world_pos']
  #   prev_position = inputs['prev|world_pos']
  #   position = 2*cur_position + acceleration - prev_position
  #   return position
