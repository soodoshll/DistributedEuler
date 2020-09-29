# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Distributed training script for supervised GraphSage.

This simple example uses two machines and each has one TensorFlow worker and ps.
Graph-learn client is colocate with TF worker, and server with ps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import graphlearn as gl
import tensorflow as tf
from graph_sage import GraphSage


# tf settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("task_index", None, "Task index")
flags.DEFINE_string("job_name", None, "worker or ps")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("tracker", '/home/ubuntu/nfs/tracker','tracker dir')

# Note: tracker dir should be cleaned up before training.
# graphlearn settings
graph_cluster = {"client_count": 2, "tracker": FLAGS.tracker, "server_count": 2}


def load_graph(config):
  dataset_folder = config['dataset_folder']
  node_type = config['node_type']
  edge_type = config['edge_type']
  g = gl.Graph() \
    .node(dataset_folder + "node_table", node_type=node_type,
          decoder=gl.Decoder(labeled=True, attr_types=["float"] * (config['features_num']), attr_delimiter=":")) \
    .edge(dataset_folder + "edge_table", edge_type=(node_type, node_type, edge_type),
          decoder=gl.Decoder(weighted=True), directed=True) \
    .node(dataset_folder + "train_table", node_type="train",
          decoder=gl.Decoder(weighted=True)) \
    .node(dataset_folder + "val_table", node_type="val",
          decoder=gl.Decoder(weighted=True)) \
    .node(dataset_folder + "test_table", node_type="test",
          decoder=gl.Decoder(weighted=True))
  return g

def train(config, graph):
  def model_fn():
    return GraphSage(graph,
                     config['class_num'],
                     config['features_num'],
                     config['batch_size'],
                     val_batch_size=config['val_batch_size'],
                     test_batch_size=config['test_batch_size'],
                     categorical_attrs_desc=config['categorical_attrs_desc'],
                     hidden_dim=config['hidden_dim'],
                     in_drop_rate=config['in_drop_rate'],
                     hops_num=config['hops_num'],
                     neighs_num=config['neighs_num'],
                     agg_type=config['agg_type'],
                     full_graph_mode=config['full_graph_mode'])

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  optimizer = gl.get_tf_optimizer(
                                 config['learning_algo'],
                                 config['learning_rate'],
                                 config['weight_decay'])
  print("opt get")
  optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=len(worker_hosts), total_num_replicas=len(worker_hosts))
  print("sync opt")
  is_chief = FLAGS.job_name == 'worker' and FLAGS.task_index == 0
  print("[{}] ischief {}".format(FLAGS.task_index, is_chief))
  sync_replicas_hook = optimizer.make_session_run_hook(is_chief, num_tokens=0)
  print("hook generated")
  trainer = gl.DistTFTrainer(model_fn,
                             cluster_spec=cluster,
                             task_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             epoch=config['epoch'],
                             optimizer=optimizer, hooks=[sync_replicas_hook])
  if FLAGS.job_name == 'worker': # also graph-learn client in this example.
    trainer.train_and_evaluate()
  else:
    trainer.join()

def main():
#  config = {'dataset_folder': '/home/ubuntu/nfs/cora/',
#            'class_num': 7,
#            'features_num': 1433,
#            'batch_size': 10, # total 140
#            'val_batch_size': 10, # total 300
#            'test_batch_size': 10, # total 1000
#            'categorical_attrs_desc': '',
#            'hidden_dim': 128,
#            'in_drop_rate': 0.5,
#            'hops_num': 2,
#            'neighs_num': [5, 2], # [25, 10]
#            'full_graph_mode': False,
#            'learning_algo': 'adam',
#            'learning_rate': 0.01,
#            'weight_decay': 0.0005,
#            'agg_type': 'gcn',
#            'epoch': 40,
#            'node_type': 'item',
#            'edge_type': 'relation'}

  config = {'dataset_folder': '/home/ubuntu/nfs/ogb-product-selfloop/', 
            'class_num': 47,
            'features_num': 100,
            'batch_size': 1000, # total 140
            'val_batch_size': 10, # total 300
            'test_batch_size': 10, # total 1000
            'categorical_attrs_desc': '',
            'hidden_dim': 256,
            'in_drop_rate': 0.5,
            'hops_num': 3,
            'neighs_num': [5, 10, 15], # [25, 10]
            'full_graph_mode': False,
            'learning_algo': 'adam',
            'learning_rate': 0.003,
            'weight_decay': 0.0005,
            'agg_type': 'mean',
            'epoch': 40,
            'node_type': 'item',
            'edge_type': 'relation'}

  g = load_graph(config)
  g_role = "server"
  if FLAGS.job_name == "worker":
    g_role = "client"
  graph_cluster = {"client_count": len(FLAGS.worker_hosts.split(",")), "tracker": FLAGS.tracker, "server_count": len(FLAGS.ps_hosts.split(","))}
  g.init(cluster=graph_cluster, job_name=g_role, task_index=FLAGS.task_index)
  train(config, g)

if __name__ == "__main__":
  main()
