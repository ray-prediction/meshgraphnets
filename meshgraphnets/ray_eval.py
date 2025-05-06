"""Functions to build evaluation metrics for cloth data."""

import tensorflow.compat.v1 as tf

from meshgraphnets.common import RayNodeType

from absl import logging

# Maybe this needs to return a set like the other eval in cloth
def evaluate(model, inputs):
    logging.info("eval called\n")
    num_prim = tf.shape(inputs['prim_vertices'])[0]
    num_tx = 1
    num_rx = tf.shape(inputs['rx_loc'])[0]
    total = num_prim + num_tx + num_rx

    mask = tf.range(total)
    mask = mask >= (total - num_rx)

    network_output = model(inputs)

    # build target total power received
    real_coeff = inputs['real_channel_coeff'][:, model.random]
    imag_coeff = inputs['imag_channel_coeff'][:, model.random]

    target_inco_sum_power = tf.math.reduce_sum(real_coeff ** 2 + imag_coeff ** 2, axis=1)
    # print(network_output[mask])
    # print(target_inco_sum_power)
    difference = target_inco_sum_power - network_output[mask]
    # print("MAPE: ", difference/target_inco_sum_power * 100)
    error = tf.reduce_sum(difference, axis=1)
    loss = tf.reduce_mean(error)
    return {
        'real_coeff': real_coeff,
        'pred_power': network_output[mask],
        'target_power': target_inco_sum_power,
        'loss': loss}


    
    
