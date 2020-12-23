import euler
import time
import argparse
from euler import Module

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("shard_idx", type=int)
parser.add_argument("shard_num", type=int)
args = parser.parse_args()
print args
if args.data == 'reddit':
    data_path = 'dataset/reddit/reddit4/'
elif args.data == "ogb-product":
    data_path = 'dataset/ogb-product/product4/'
elif args.data == "wn18":
    data_path = 'dataset/WN18/wn18-4/'
else:
    print "unknown dataset"
    exit(-1)

euler.start(
        directory=data_path, 
        shard_idx=args.shard_idx,    
        shard_num=args.shard_num,   
        zk_addr='node0:2181',
        zk_path='/euler', 
        module=Module.DEFAULT_MODULE | Module.EDGE)
while True:
        time.sleep(1)
