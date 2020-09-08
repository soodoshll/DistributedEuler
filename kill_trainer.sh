#!/bin/bash
for i in {1..3}
do
       ssh node$i "pkill -f -9 run_graphsage.py" 
done
pkill -f -9 run_graphsage.py


for i in {0..3}
do
        ssh node$i "rm -rf ~/euler_profile/ckpt"
done
