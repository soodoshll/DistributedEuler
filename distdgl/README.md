This is the benchmark script of DistDGL.

## Usage

Follow the instructions in `DistDGL/examples/pytorch/graphsage/experimental` and copy(or link) these files into the workspace directory.

### Log format

We need to modify the training script (like `train_dist.py`) to write logs into files. Log files with a path like `<workspace>/log_*.txt` will be downloaded after the experiment. Current naming convention is like:

 * `log_itr_<trainer_id>.txt`: log of each iteration, like iteration time. The first line is the table head.
 * `log_epoch_<trainers_id>.txt`: log of each epoch, like the epoch time.

`train_dist.py` script in this repo is an example.

### Run experiments

* Modify `experiment.py` to match with your own setting, like the `workspace` directory and the experiment command.
* You may need to manually set `OMP_NUM_THREADS` in the launch script.

Run experiments in one line (under the `<workspace>` directory):
```
python experiment.py
```

The script will create a log directory with the timestamp as its name under `<workspace>/log`, and a symbol link `<workspace>/log/latest` linked to this directory. So to get the result of the latest experiment, just enter `<workspace>/log/latest`.

You can also add your own experiment in it.

If the program dies and you want to restart the experiment but not to create a new directory, you can use
```
python experiment.py -c
```

### Data Analysis

```
python analyze_log.py
```

or

```
python analyze_log.py <experiment>
```

it will generate a report of the experiment according to the log files in `<workspace>/log/<experiment>`.

To watch the result during experiment running:

```
watch -n 10 python analyze_log.py
```