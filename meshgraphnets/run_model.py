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
"""Runs the learner/evaluator."""

import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import ray_eval
from meshgraphnets import ray_model
from meshgraphnets import core_model
from meshgraphnets import dataset

# from memory_profiler import profile

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth', 'ray'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval),
    'ray': dict(noise=0.003, gamma=0.1, field='world_pos', history=True, # TODO: this needs specific tuning
                  size=1, batch=1, model=ray_model, evaluator=ray_eval),  
}

# @profile
def learner(model, params):
  """Run a learner job."""
  if FLAGS.model == 'ray':
    ds = dataset.load_ray_dataset(FLAGS.dataset_dir, 'train')
  else:  
    ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
    ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])

  # TODO: I really should add noise to the tx/rx positions!!!
  # ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
  #                                   noise_scale=params['noise'],
  #                                   noise_gamma=params['gamma'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()

  loss_op = model.loss(inputs)
  global_step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(learning_rate=1e-4,
                                  global_step=global_step,
                                  decay_steps=int(5e6),
                                  decay_rate=0.1) + 1e-6
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step)
  # Don't train for the first few steps, just accumulate normalization stats
  train_op = tf.cond(tf.less(global_step, 100),
                     lambda: tf.group(tf.assign_add(global_step, 1)),
                     lambda: tf.group(train_op))

  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=600) as sess:

    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])
      # if step == 2:
      #   return
      if step % 1 == 0:
        logging.info('Step %d: Loss %g', step, loss)
    logging.info('Training complete.')


def evaluator(model, params):
  """Run a model rollout trajectory."""
  if FLAGS.model == 'ray':
    ds = dataset.load_ray_dataset(FLAGS.dataset_dir, 'train')
  else:
    ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
    ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    # trajectories = []
    scalars = []
    # for traj_idx in range(FLAGS.num_rollouts):
    #   logging.info('Rollout trajectory %d', traj_idx)
    scalar_data = sess.run([scalar_op])
    scalars.append(scalar_data)
    for l in scalars:
      print(l)    
    # for key in scalars[0]:
    #   logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    # with open(FLAGS.rollout_path, 'wb') as fp:
    #   pickle.dump(trajectories, fp)

# @profile
def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  # tf.enable_eager_execution()

  params = PARAMETERS[FLAGS.model]
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=15)
  print(params['model'])
  model = params['model'].Model(learned_model)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  app.run(main)
