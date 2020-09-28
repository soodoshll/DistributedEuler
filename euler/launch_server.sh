#!/bin/bash
for i in {0..3}
do
        ssh node$i "cd ~/euler_profile; OMP_NUM_THREADS=1 python start_euler.py $1 $i 4 >server.out 2>server.err &"
done
