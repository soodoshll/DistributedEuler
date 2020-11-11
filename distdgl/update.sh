#!/bin/bash
WORKSPACE=~/graphsage
num_remote=`wc -l ip_config.txt | cut -d " " -f1`
echo Number of remote machines: $num_remote
if [ -f "port.txt" ]; then
        read last_port < port.txt;
else
        last_port=9345;
fi
port=`expr $last_port + 200`
if [ $port -gt 50000 ]
then
        port=9345;
fi
echo 'choose port' $port
echo $port > port.txt
awk '{print $1,'${port}' > "ip_config.txt"}'  ip_config.txt
local_ip=`hostname -i`
echo local ip $local_ip
while IFS="" read -r line || [ -n "$line" ]
do
    remote=`echo "$line" | cut -d " " -f1`
    if [ $remote == $local_ip ]
    then
        continue
    fi
    echo $remote
    ssh $remote "pkill -f train_dist ; rm /dev/shm/*"   </dev/null
	scp entity_classify_dist.py train_dist_unsupervised.py train_dist.py train_GAT.py ip_config.txt $remote:$WORKSPACE/ &
done <ip_config.txt
wait
pkill -f train_dist
pkill -f torch
rm /dev/shm/* 
