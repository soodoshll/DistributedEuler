This repo contains some scripts for running synchronous distributed graphsage training using graph-learn(AliGraph).

### Usage

+ copy `trainer.py` to the `graphlearn/python/model/tf/trainer.py` directory in where you installed graph-learn.
+ copy data and scripts to remote machines. Or use NFS to store data
+ run `launch.py` to launch distributed training. Example:
```
python3 launch.py "python dist_train.py"    
```

Unfortunately, because the launch script is written in python 3 while graph-learn only support python2, we have to write in such a strange way.
