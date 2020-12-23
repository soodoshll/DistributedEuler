from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print("Sayonara Sayonara")
import tensorflow as tf
import tf_euler
import json
import os
from euler_estimator import NodeEstimator
from graphsage import SupervisedGraphSage

print("ev set by user:", os.environ.keys())
os.environ["OMP_NUM_THREADS"]="1" 

class Estimator(NodeEstimator):
    def __init__(self, model_class, params, run_config, num_worker):
        super(Estimator, self).__init__(model_class, params, run_config)
        self.num_worker = num_worker
    def train_model_init(self, model, features, mode, params):
        source = self.get_train_from_input(features, params)
        _, loss, metric_name, metric = model(source)
        global_step = tf.train.get_or_create_global_step()
        opt = tf_euler.utils.optimizers.get(
            params.get('optimizer', 'adam'))(
            params.get('learning_rate', 0.001))
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=self.num_worker, total_num_replicas=self.num_worker)
        train_op = opt.minimize(loss, global_step)
        print("!!###@@@", tf.flags.FLAGS.task_type, tf.flags.FLAGS.task_type == 'chief')
        sync_replicas_hook = opt.make_session_run_hook(tf.flags.FLAGS.task_type == 'chief', num_tokens=0)    
        hooks = []
        hooks.append(sync_replicas_hook)
        tensor_to_log = {'step': global_step,
            'loss': loss,
            metric_name: metric}
        hooks.append(
            tf.train.LoggingTensorHook(
            tensor_to_log, every_n_iter=params.get('log_steps', 100)))
        spec = tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=hooks)
        return spec

def define_network_flags():
    tf.flags.DEFINE_string('dataset', 'reddit', '')
    tf.flags.DEFINE_integer('shard_num', 4, 'Number of shards')
    tf.flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension.')
    tf.flags.DEFINE_integer('layers', 2, 'SAGE convolution layer number.')
    tf.flags.DEFINE_list('fanouts', [10, 25], 'GraphSage fanouts.')
    tf.flags.DEFINE_integer('batch_size', 32, 'Mini-batch size')
    tf.flags.DEFINE_integer('num_epochs', 10, 'Epochs to train')
    tf.flags.DEFINE_integer('log_steps', 20, 'Number of steps to print log.')
    tf.flags.DEFINE_string('model_dir', 'ckpt', 'Model checkpoint.')
    tf.flags.DEFINE_string('task_type', 'worker', '')
    tf.flags.DEFINE_integer('task_id', 0, '')
    tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
    tf.flags.DEFINE_enum('optimizer', 'adam', ['adam', 'adagrad', 'sgd', 'momentum'],
                         'Optimizer algorithm')
    tf.flags.DEFINE_enum('run_mode', 'train', ['train', 'evaluate', 'infer'],
                         'Run mode.')
    tf.flags.DEFINE_string('chief', '', '')
    tf.flags.DEFINE_string('ps', '', '')
    tf.flags.DEFINE_string('worker', '', '')


def get_dataset(dataset):
    if dataset == 'reddit':
        return {"train_label" : ['train'],
                "all_label" : ['train', 'train_removed'],
                "max_node_id" : 231443,
                "feature_dim" : 602,
                "num_classes" : 41
                }
    elif dataset == 'ogb-product':
        return {"train_label" : [0],
                "all_label" : [0],
                "max_node_id" : 2449029,
                "feature_dim" : 100,
                "num_classes" : 47
                }
        

def main(_):
    print("Kai Jun Jian")
    print("ev set by user:", os.environ.keys())
    flags_obj = tf.flags.FLAGS
    cluster = {'chief': flags_obj.chief.split(','),
               'ps': flags_obj.ps.split(','),
               'worker': flags_obj.worker.split(',')}
    task_type = flags_obj.task_type 
    task_id = flags_obj.task_id
    os.environ['TF_CONFIG'] = json.dumps(
                {'cluster': cluster,
                 'task': {'type': task_type, 'index': task_id}
                })

    tf.logging.set_verbosity(tf.logging.INFO)
    if flags_obj.task_type != "ps":
        tf_euler.initialize_graph({
            'mode': 'remote',
            'zk_server': 'node0:2181',
            'zk_path': '/euler',
            'shard_num': flags_obj.shard_num,
            'num_retries': 1
        })
    # print("HERE")

    dataset = get_dataset(flags_obj.dataset)

    fanouts = list(map(int, flags_obj.fanouts))
    assert flags_obj.layers == len(fanouts)
    dims = [flags_obj.hidden_dim] * (flags_obj.layers + 1)
    if flags_obj.run_mode == 'train':
        metapath = [dataset['train_label']] * flags_obj.layers
    else:
        metapath = [dataset['all_label']] * flags_obj.layers
    num_steps = int((dataset['max_node_id'] + 1) // flags_obj.batch_size *
                    flags_obj.num_epochs)

    model = SupervisedGraphSage(dims, fanouts, metapath,
                                'feature',
                                dataset['feature_dim'],
                                'label',
                                dataset['num_classes'],)
    params = {'train_node_type': 'train', 
              'batch_size': flags_obj.batch_size,
              'optimizer': flags_obj.optimizer,
              'learning_rate': flags_obj.learning_rate,
              'log_steps': flags_obj.log_steps,
              'model_dir': flags_obj.model_dir,
              'id_file': '',
              'infer_dir': flags_obj.model_dir,
              'total_step': num_steps,
              'save_checkpoints_steps': None,
              'save_checkpoints_secs': None}
    #opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    #opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=2)


    sess_config = tf.ConfigProto(allow_soft_placement=True, device_count={"CPU": 1}, intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
    sess_config.intra_op_parallelism_threads = 1
    sess_config.inter_op_parallelism_threads = 1
    config = tf.estimator.RunConfig(log_step_count_steps=None, session_config=sess_config)
    num_worker = len(cluster['chief']) + len(cluster['worker']) 
    model_estimator = Estimator(model, params, config, num_worker)

    if flags_obj.run_mode == 'train':
    #    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir_'+str(flags_obj.task_type)+"_"+str(flags_obj.task_id), dump_steps=[10]) as pctx:
        model_estimator.train_and_evaluate()
    #elif flags_obj.run_mode == 'evaluate':
    #    model_estimator.evaluate()
    elif flags_obj.run_mode == 'infer':
        model_estimator.infer()
    else:
        raise ValueError('Run mode not exist!')


if __name__ == '__main__':
    print("Hen You Jing Shen")
    tf.logging.set_verbosity(tf.logging.INFO)
    define_network_flags()
    tf.app.run(main)
