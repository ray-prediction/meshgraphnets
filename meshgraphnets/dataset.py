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
"""Utility functions for reading the datasets."""

import functools
import json
import os

import tensorflow.compat.v1 as tf

from meshgraphnets.common import NodeType

from absl import logging


def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out

feature_description = {
    'scene_adj_flat': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'scene_adj_lengths': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'tx_adj_flat': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'tx_adj_lengths': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'rx_adj_flat': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'rx_adj_lengths': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'tx_loc': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'num_tx': tf.io.FixedLenFeature([1], tf.int64),
    'rx_loc': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'num_rx': tf.io.FixedLenFeature([1], tf.int64),
    'prim_vertices': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'prim_normals': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'num_prim': tf.io.FixedLenFeature([1], tf.int64),
    'real_channel_coeff': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'imag_channel_coeff': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
}

def _parse_ray(example_proto):
    # print('called _parse')
    logging.info('called _parse')
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Reshape based on metadata
    tx_loc = tf.reshape(parsed['tx_loc'], [parsed['num_tx'][0], 3])
    rx_loc = tf.reshape(parsed['rx_loc'], [parsed['num_rx'][0], 3])
    prim_vertices = tf.reshape(parsed['prim_vertices'], [parsed['num_prim'][0], 3, 3])  # assuming triangles with 3 vertices
    prim_normals = tf.reshape(parsed['prim_normals'], [parsed['num_prim'][0], 3])
    real_channel_coeff = tf.reshape(parsed['real_channel_coeff'], [parsed['num_rx'][0], parsed['num_tx'][0], -1])
    imag_channel_coeff = tf.reshape(parsed['imag_channel_coeff'], [parsed['num_rx'][0], parsed['num_tx'][0], -1])

    scene_adj = tf.RaggedTensor.from_row_lengths(parsed['scene_adj_flat'], parsed['scene_adj_lengths'])
    tx_adj = tf.RaggedTensor.from_row_lengths(parsed['tx_adj_flat'], parsed['tx_adj_lengths'])
    rx_adj = tf.RaggedTensor.from_row_lengths(parsed['rx_adj_flat'], parsed['rx_adj_lengths'])

    scene_adj = tf.cast(scene_adj, tf.int32)
    tx_adj = tf.cast(tx_adj, tf.int32)
    rx_adj = tf.cast(rx_adj, tf.int32)

    return {
        'scene_adj': scene_adj,
        'tx_adj': tx_adj,
        'rx_adj': rx_adj,
        'tx_loc': tx_loc,
        'rx_loc': rx_loc,
        'prim_vertices': prim_vertices,
        'prim_normals': prim_normals,
        'real_channel_coeff': real_channel_coeff,
        'imag_channel_coeff': imag_channel_coeff,
    }


def load_ray_dataset(path, split):
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse_ray), num_parallel_calls=1)
  # ds = ds.prefetch(1)
  return ds

def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


def add_targets(ds, fields, add_history):
  """Adds target and optionally history fields to dataframe."""
  def fn(trajectory):
    out = {}
    for key, val in trajectory.items():
      out[key] = val[1:-1]
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
        out['target|'+key] = val[2:]
    return out
  return ds.map(fn, num_parallel_calls=8)


def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma):
  """Splits trajectories into frames, and adds training noise."""
  def add_noise(frame):
    noise = tf.random.normal(tf.shape(frame[noise_field]),
                             stddev=noise_scale, dtype=tf.float32)
    # don't apply noise to boundary nodes
    mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
    noise = tf.where(mask, noise, tf.zeros_like(noise))
    frame[noise_field] += noise
    frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
    return frame

  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  ds = ds.map(add_noise, num_parallel_calls=8)
  ds = ds.shuffle(10000)
  ds = ds.repeat(None)
  return ds.prefetch(10)


def batch_dataset(ds, batch_size):
  """Batches input datasets."""
  shapes = ds.output_shapes
  types = ds.output_types
  def renumber(buffer, frame):
    nodes, cells = buffer
    new_nodes, new_cells = frame
    return nodes + new_nodes, tf.concat([cells, new_cells+nodes], axis=0)

  def batch_accumulate(ds_window):
    out = {}
    for key, ds_val in ds_window.items():
      initial = tf.zeros((0, shapes[key][1]), dtype=types[key])
      if key == 'cells':
        # renumber node indices in cells
        num_nodes = ds_window['node_type'].map(lambda x: tf.shape(x)[0])
        cells = tf.data.Dataset.zip((num_nodes, ds_val))
        initial = (tf.constant(0, tf.int32), initial)
        _, out[key] = cells.reduce(initial, renumber)
      else:
        merge = lambda prev, cur: tf.concat([prev, cur], axis=0)
        out[key] = ds_val.reduce(initial, merge)
    return out

  if batch_size > 1:
    ds = ds.window(batch_size, drop_remainder=True)
    ds = ds.map(batch_accumulate, num_parallel_calls=8)
  return ds
