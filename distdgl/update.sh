#!/bin/bash
WORKSPACE=$1
num_remote=`wc -l ip_config.txt | cut -d " " -f1`
echo Number of remote machines: $num_remote
if [ -f "port.txt" ]; then
        read last_port < port.txt;
else
        last_port=40000;
fi
port=`expr $last_port + 345`
if [ $port -gt 60000 ]
then
        port=40000;
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
    ssh -n $remote "pkill -f train ; pkill -f python ; rm /dev/shm/*"
    # Is there any elegant way to kill processes?
    scp ip_config.txt $remote:$WORKSPACE/
done <ip_config.txt
wait
pkill -f train
pkill -f torch
pkill -f multiprocessing
rm /dev/shm/* 
