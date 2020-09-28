from ogb.nodeproppred import NodePropPredDataset
import sys
import numpy as np
from tqdm import tqdm

data_name = sys.argv[1]
dataset = NodePropPredDataset(name = data_name)

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0] # graph: library-agnostic graph object


num_classes = np.max(label) + 1
num_nodes = graph["num_nodes"]

node_buf = []

train_mask = np.zeros(num_nodes)
train_mask[train_idx] = 1

valid_mask = np.zeros(num_nodes)
valid_mask[valid_idx] = 1

test_mask = np.zeros(num_nodes)
test_mask[test_idx] = 1

print("converting nodes...")
for nid in tqdm(range(num_nodes)):
    node = {"id" : nid, "weight" : 1.0}
    if train_mask[nid]:
        node_type = "train"
    elif valid_mask[nid]:
        node_type = "valid"
    else:
        node_type = "test"
    node['type'] = node_type
    feat = graph['node_feat'][nid].tolist()
    node['features'] = [{"name":"feature", "type":"dense", "value":feat}]
    node_buf.append(node)

print("attatching labels...")
for i in tqdm(range(len(train_idx))):
    onehot = np.zeros(num_classes)
    l = label[i]
    onehot[l] = 1
    onehot = onehot.tolist()
    nid = train_idx[i]
    node_buf[nid]['features'].append({"name":"label", "type":"dense", "value":onehot})


print("converting edges...")
edge_buf = []
src, dst = graph['edge_index']
num_edges = src.shape[0]
cnt = 0
for i in tqdm(range(num_edges)):
    s = src[i]
    d = dst[i]
    edge_buf.append({
        "src":s.tolist(),
        "dst":d.tolist(),
        "type":0,
        "weight":1.0,
        "features":[]
        })

buf = {"nodes":node_buf, "edges":edge_buf}

print("writing to file...")
import json
json.dump(buf, open(data_name+".json", "w"))
