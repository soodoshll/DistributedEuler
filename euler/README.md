This repo contains some scripts for running distributed graphsage on Euler(https://github.com/alibaba/euler/) since its original repo lacks some important guide of distributed training and part of its doc is outdated. (In fact, the Chinese doc is for v2.0 and the English doc is still for 1.0. And almost all issues are in Chinese.)

To run these code, you need to modify them according to your cluster configuration.

### Usage

 + `ogb_dataset.py`: to convert ogb dataset into euler json format. Note that you still need to convert the generated json file into euler binary format. To convert the ogbn-products graph on a 96 vCPU AWS EC2 instance, it takes about 30min to convert original data into json file and takes more than 3 hours to convert json file into binary, while it will occupy more than 200G memory.
 + `launch_server.sh` : to launch euler servers. Note that you need to first start the zookeeper server. The single node mode is sufficient.
 + `lanuch.py` : to start training. This launch scripted is adapted from that of DGL. 

### Result

    


