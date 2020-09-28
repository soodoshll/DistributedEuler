#!/bin/bash
for i in {1..3}
do
        ssh node$i "pkill -f -9 start_euler.py"
done 
pkill -f -9 start_euler
