#!/bin/bash
for i in {1..3}
do
        scp ~/euler_profile/*.py node$i:~/euler_profile/
done
