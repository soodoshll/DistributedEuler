import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.distributed import DistDataLoader

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision.models as models
#import torch.autograd.profiler as profiler
from pyinstrument import Profiler

def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors, device):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.device = device
        self.cache = None

    def sample_blocks(self, seeds):
        try:
            seeds = th.LongTensor(np.asarray(seeds))
            blocks = []
            if self.cache is None:
                for fanout in self.fanouts:
                    # For each seed node, sample ``fanout`` neighbors.
                    frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
                    # Then we compact the frontier into a bipartite graph for message passing.
                    block = dgl.to_block(frontier, seeds)
                    # Obtain the seed nodes for next layer.
                    seeds = block.srcdata[dgl.NID]

                    blocks.insert(0, block)
                #self.cache = blocks
            else:
                blocks = self.cache

            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]
            #batch_inputs = th.zeros((len(input_nodes), 128), dtype=th.float32)
            #batch_labels = th.zeros(len(seeds), dtype=th.long)
            batch_inputs, batch_labels = load_subtensor(self.g, seeds, input_nodes, "cpu")
            blocks[0].srcdata['features'] = batch_inputs
            blocks[-1].dstdata['labels'] = batch_labels
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
        return blocks

class DistSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(np.arange(g.number_of_nodes()),
                                           g.get_partition_book(), force_even=True)
        y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_hidden), th.float32, 'h',
                                       persistent=True)
        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_classes),
                                               th.float32, 'h_last', persistent=True)

            sampler = NeighborSampler(g, [-1], dgl.distributed.sample_neighbors, device)
            print('|V|={}, eval batch size: {}'.format(g.number_of_nodes(), batch_size))
            # Create PyTorch DataLoader for constructing blocks
            dataloader = DistDataLoader(
                dataset=nodes,
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=False,
                drop_last=False)

            for blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                input_nodes = block.srcdata[dgl.NID]
                output_nodes = block.dstdata[dgl.NID]
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            #g.barrier()
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.distributed.sample_neighbors, device)

    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=train_nid.numpy(),#[:args.batch_size],
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        #queue_size=16
        )

    train_nid_full = th.nonzero(g.ndata['train_mask'][0:g.number_of_nodes()], as_tuple=True)[0]

    # Define model and optimizer
    model = DistSAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_size = th.sum(g.ndata['train_mask'][0:g.number_of_nodes()])
    weight = th.ones([th.distributed.get_world_size()])

    # Training loop
    iter_tput = []
    profiler = Profiler()
    if args.close_profiler == False:
        profiler.start()
    epoch = 0
    itr_logfile = open("log_itr_{}.txt".format(g.rank()),'w')
    epoch_logfile = open("log_epoch_{}.txt".format(g.rank()),'w')
    epoch_time = []

    print("itr\tsample\tcopy\tforward\tbackward\tupdate\tinput\tedges\tseeds", file=itr_logfile)
    for epoch in range(args.num_epochs):

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        step_time = []
        sample_time = []
        copy_time = []
        forward_time = []
        backward_time = []
        update_time = []
        itr_time = []
        num_inputs = []
        num_seeds = []
        num_edges = []
        tic = time.time()
        start = time.time()
        itr_start = time.time()
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()
            sample_time.append(tic_step - itr_start)

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            th.cuda.synchronize()
            #g.barrier()
            start = time.time()
            num_seeds.append(len(blocks[-1].dstdata[dgl.NID]))
            num_inputs.append(len(blocks[0].srcdata[dgl.NID]))
            num_edges.append(blocks[0].number_of_edges())
            blocks = [block.to(device) for block in blocks]
            #batch_inputs = th.zeros((len(blocks[0].srcdata[dgl.NID]), 128), dtype=th.float32, device=device)
            #batch_labels = th.zeros(len(blocks[-1].dstdata[dgl.NID]), dtype=th.long, device=device) 
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']
            batch_labels = batch_labels.long()
            th.cuda.synchronize()
            copy_time.append(time.time() - start)
            #g.barrier()
            # Compute loss and prediction
            th.cuda.synchronize()
            start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            th.cuda.synchronize()
            forward_end = time.time()
            #g.barrier()
            th.cuda.synchronize()
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            th.cuda.synchronize()
            compute_end = time.time()
            forward_time.append(forward_end - start)
            backward_time.append(compute_end - backward_start)

            optimizer.step()
            th.cuda.synchronize()
            update_time.append(time.time() - compute_end)

            #g.barrier()
            itr_time.append(time.time() - itr_start)
            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
            #print(dataloader.queue.qsize())
            itr_start = time.time()

        toc = time.time()
        if epoch >=3:
            epoch_time.append(toc-tic)
        for itr in zip(itr_time, sample_time, copy_time, forward_time, backward_time, update_time, num_inputs, num_edges, num_seeds):
            print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}".format(*itr), file=itr_logfile)
        print("[{}] Epoch {:.4f} | Sample {:.4f} | CPU-GPU copy {:.4f} | Forward {:.4f} | Backward {:.4f} | Update {:.4f} | #inputs {:.1f} | #edges {:.1f} | #seeds {:.1f}".format(g.rank(), toc-tic, sum(sample_time), sum(copy_time), sum(forward_time), sum(backward_time), sum(update_time), np.mean(num_inputs), np.mean(num_edges), np.mean(num_seeds)), file=epoch_logfile)

        if args.dynamic_batch and epoch < 3:
            record = [th.Tensor([0.0]) for _ in range(th.distributed.get_world_size())]
            #record_local = np.mean(num_inputs)
            record_local = sum(sample_time) + sum(copy_time) + 2 * sum(forward_time)
            th.distributed.all_gather(record, th.Tensor([record_local]))
            # After an epoch, re-split the training set
            record = th.cat(record)
            record_inv = 1 / record
            w = record_inv / th.mean(record_inv)
            a = 0.8
            w = a * w + (1 - a)
            weight = weight * w
        
            # Split to machines
            pb = g.get_partition_book()
            part_weight = th.reshape(weight, (pb.num_partitions(), -1))
            part_weight = th.mean(part_weight, dim=1).numpy()
            batch_size = part_weight[pb.partid] * args.batch_size
            batch_size = int(batch_size)
            train_nid = dgl.distributed.node_split(g.ndata['train_mask'], pb, force_even=True, part_weight=part_weight)
            
            # Check the number of batches is the same.
            batch_num = th.tensor([len(train_nid) / batch_size], dtype=th.float) 
            batch_num_gather = [th.tensor([0], dtype=th.float) for _ in range(th.distributed.get_world_size())]
            th.distributed.all_gather(batch_num_gather, batch_num)
            batch_num_gather = th.cat(batch_num_gather)
            batch_num_int = batch_num_gather.long()
            batch_num_min = th.min(batch_num_int)
            if (batch_num_int[g.rank()] > batch_num_min):
                # I'm not sure
                # batch_size = len(train_nid) // batch_num_min.numpy()
                batch_size += 1

            print("Part {}, seeds {}, batch size {}, batch {}".format(g.rank(), len(train_nid), batch_size, len(train_nid) / batch_size))

            dataloader = DistDataLoader(
                dataset=train_nid.numpy(),#[:args.batch_size],
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=True,
                drop_last=False)
            g.barrier()
   
    print("Average Epoch Time: {}".format(np.mean(epoch_time)), file=epoch_logfile)
    if args.close_profiler == False :
        profiler.stop()
        #if g.rank() % 4 == 0:
        print(profiler.output_text(unicode=True, color=True, show_all=True))
    itr_logfile.close()
    epoch_logfile.close()

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    if not args.standalone:
        th.distributed.init_process_group(backend='gloo')
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print('rank:', g.rank())

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata['train_mask'], pb, force_even=True)
    val_nid = dgl.distributed.node_split(g.ndata['val_mask'], pb, force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata['test_mask'], pb, force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print("[{}], train nodes {}, batchsize {}, batches {:.2f}".format(g.rank(), len(train_nid), args.batch_size, len(train_nid) / args.batch_size))
    print('part {}, train: {} (local: {}), val: {} (local: {}), test: {} (local: {})'.format(
        g.rank(), len(train_nid), len(np.intersect1d(train_nid.numpy(), local_nid)),
        len(val_nid), len(np.intersect1d(val_nid.numpy(), local_nid)),
        len(test_nid), len(np.intersect1d(test_nid.numpy(), local_nid))))
    if args.num_gpus == -1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:'+str(g.rank() % args.num_gpus))
    labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print('#labels:', n_classes)
    # Pack data
    in_feats = g.ndata['features'].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_clients', type=int, help='The number of clients')
    parser.add_argument('--num_servers', type=int, default=1, help='The number of servers')
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument('--num_gpus', type=int, default=-1, 
                        help="the number of GPU device. Use -1 for CPU training")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_hidden', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--fan_out', type=str, default='10,25')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_eval', type=int, default=100000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    parser.add_argument('--close_profiler', action='store_true', help='Close pyinstrument profiler')
    parser.add_argument('--dynamic_batch', action='store_true')
    args = parser.parse_args()
    assert args.num_workers == int(os.environ.get('DGL_NUM_SAMPLER')), \
    'The num_workers should be the same value with DGL_NUM_SAMPLER.'
    assert args.num_servers == int(os.environ.get('DGL_NUM_SERVER')), \
    'The num_servers should be the same value with DGL_NUM_SERVER.'

    print(args)
    main(args)
