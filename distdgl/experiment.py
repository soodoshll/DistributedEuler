import os
import time
import subprocess
import itertools

# create log directory and add symbolic link
LOG_DIR_PREFIX = "log"
LATEST_LOG_DIR = "log/latest"

# read ip config
with open("ip_config.txt", "r") as f:
  machine_ip = [line.split(' ')[0] for line in f]

def create_log_dir():
  time_stamp = time.strftime("%m%d-%H%M%S", time.localtime())
  log_dir = os.path.join(LOG_DIR_PREFIX, time_stamp)
  os.makedirs(log_dir, exist_ok=True)
  log_dir = os.path.abspath(log_dir)
  if os.path.lexists(LATEST_LOG_DIR):
    print("LOG dir already exists")
    os.remove(LATEST_LOG_DIR)
  os.symlink(log_dir, LATEST_LOG_DIR)
  print("Create log directory:", log_dir)

def run_experiment(cmd, name='', timeout=1800):
  print("Cleaning environment...")
  os.system("./update.sh")
  print("Cleaning environment finished, start running")
  try:
    subprocess.run(cmd, timeout=timeout, shell=True)
  except subprocess.TimeoutExpired:
    print("TIME OUT")
    os.system("./update.sh")

  if name == '':
    name = time.strftime("%m%d-%H%M%S", time.localtime())
  log_dir = os.path.join(LATEST_LOG_DIR, name)
  os.mkdir(log_dir)
  # record command
  with open(os.path.join(log_dir, "cmd.txt"), "w") as f:
    print(cmd, file=f)

  # gather log
  os.system("./gather_and_clean.sh " + log_dir)


def run_graphsage():
  num_gpu_per_machine = [1]
  fanouts = [5, 10]
  hops = [3]
  batch = [1000, 2000, 5000]
  for n_g, f, h, b in itertools.product(num_gpu_per_machine, fanouts, hops, batch):
    f_multilayer = ','.join([str(f)] * h)
    run_experiment('python3 launch.py --workspace ~/graphsage/ --num_trainers {} --num_samplers 8 --num_servers 8 --part_config ogb-product/ogb-product.json --ip_config ip_config.txt "python3 train_dist.py --graph_name ogb-product --ip_config ip_config.txt --num_servers 8 --num_epochs 1 --batch_size {}  --num_workers 8 --num_gpus {} --num_hidden 128 --fan_out {} --num_layers {} --eval_every 9999 --close_profiler"'.format(n_g, b, n_g, f_multilayer, h), name="{}_{}_{}_{}".format(n_g, f, h, b))

def run_GAT():
  pass

if __name__ == "__main__":
  create_log_dir()
  run_graphsage()
  os.system("./update.sh")
