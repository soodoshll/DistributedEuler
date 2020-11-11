This is the benchmark script of DistDGL.

## Usage

Follow the instructions in `DistDGL/examples/pytorch/graphsage/experimental` and copy these files into the workspace directory.

### Log format

We need to modify the training script (like `train_dist.py`) to write logs into files. Log files with a path like `<workspace>/log_*.txt` will be downloaded after the experiment. Current naming convention is like:

 * `log_itr_<trainer_id>.txt`: log of each iteration, like iteration time. The first line is the table head.
 * `log_epoch_<trainers_id>.txt`: log of each epoch, like the epoch time.

`train_dist.py` script in this repo is an example.

### Run experiments

Run experiments in one line:
```
python experiment.py
```

The script will automatically create a log directory with the timestamp as its name under `<workspace>/log`, and a symbol link `<workspace>/log/latest` linked to this directory. So to get the result of the latest experiment, just enter `<workspace>/log/latest`.

### Data Analysis

```
python analyze_log.py
```