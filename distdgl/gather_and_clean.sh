#!/bin/bash
WORKSPACE=~/graphsage
num_remote=`wc -l ip_config.txt | cut -d " " -f1`
echo Number of remote machines: $num_remote
while IFS="" read -r line || [ -n "$line" ]
do
  remote=`echo "$line" | cut -d " " -f1`
  echo $remote
	scp $remote:$WORKSPACE/log_\*.txt $1
  ssh -n $remote rm $WORKSPACE/log_\*.txt
done <ip_config.txt
wait
