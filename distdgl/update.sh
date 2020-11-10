#!/bin/bash
WORKSPACE=~/graphsage
num_remote=`wc -l ip_config.txt | cut -d " " -f1`
echo Number of remote machines: $num_remote
port=`expr $RANDOM + 10000`
echo 'choose port' $port
awk '{print $1,'${port}' > "ip_config.txt"}'  ip_config.txt
local_ip=`hostname -i`
echo local ip $local_ip
while IFS="" read -r line || [ -n "$line" ]
do
    remote=`echo "$line" | cut -d " " -f1`
    echo $remote
    if [ $remote == $local_ip ]
    then
        continue
    fi
    ssh -n $remote "pkill -f -9 train ; rm /dev/shm/*"  &
	scp entity_classify_dist.py train_dist_unsupervised.py train_dist.py train_GAT.py ip_config.txt $remote:$WORKSPACE/ &
done <ip_config.txt
wait
pkill -f -9 train
rm /dev/shm/* 
