import os
import sys
import csv
import numpy as np
LOG_PATH = "./log"

def read_epoch_time(log_dir):
  experiments = os.listdir(log_dir)
  experiments.sort()
  print("========== Average epoch time(s) ==========")
  for e in experiments:
    exp_dir = os.path.join(log_dir, e)
    epoch_time = None

    # read the data of the first trainer
    log_path = os.path.join(exp_dir, "log_epoch_0.txt")
    if os.path.exists(log_path):
      try:
        with open(log_path, "r") as f:
          ret = f.readlines()[-1].strip().split()[-1]
        epoch_time = ret
        print("{}\t{}".format(e, epoch_time))
      except IndexError:
        print("{}\tN/A".format(e))

def read_itr_log(path):
  with open(path) as f:
    title = f.readline()
    title = title.strip().split('\t')
    reader = csv.reader(f, delimiter='\t')
    rows = [list(map(float, x)) for x in reader]
  return title, np.array(rows)

def read_itr_log_exp(path):
  num_trainers = len(log_files(path)) // 2
  title = None
  data = []
  for i in range(num_trainers):
    itr_log = "log_itr_{}.txt".format(i)
    itr_log = os.path.join(path, itr_log)
    if not os.path.exists(itr_log):
      title = None
      break
    title, rows = read_itr_log(itr_log)
    data.append(rows)
  data = np.array(data)
  return title, data

def log_files(path):
  return list(filter(lambda x: x[:3]=='log' and x[-4:]=='.txt', os.listdir(path)))

def components(path):
  experiments = os.listdir(log_dir)
  experiments.sort()
  print("========== Time spent on each component (per iteration) (s) ==========")
  for e in experiments:
    exp_dir = os.path.join(log_dir, e)
    title, data = read_itr_log_exp(exp_dir)
    if title is None or len(data.shape) < 3:
      print("{}\t N/A".format(e))
      continue
    try:
      avg = np.mean(data, axis=1)
      avg = np.mean(avg, axis=0)
      print(e,"\t", end="")
      for field, value in zip(title, avg):
        print("{}: {:.3f} | ".format(field, value), end="")
      print("")
    except ValueError:
      print("{}\tN/A".format(e))

def diff_backward_time(path):
  experiments = os.listdir(log_dir)
  experiments.sort()
  print("========== difference of backward time / iteration time(s) ==========")
  for e in experiments:
    exp_dir = os.path.join(log_dir, e)
    num_trainers = len(os.listdir(exp_dir)) // 2
    data = []
    title, data = read_itr_log_exp(exp_dir)
    if title is None or data.shape[0] < num_trainers or num_trainers <= 0:
      print("{}\t N/A data incomplete".format(e))
      continue
    try:
      backward_idx = title.index('backward')
      itr_idx = title.index('itr')
      backward = data[:, :, backward_idx]
      itr_time = data[:, :, itr_idx]
      max_bw = np.max(backward, axis=0)
      min_bw = np.min(backward, axis=0)
      print("{}\t{:.4f}\t{:.4f}".format(e, np.mean(max_bw) - np.mean(min_bw), np.mean(itr_time))) # [trainer, iteration, field]
    except (ValueError, IndexError):
      print("{}\tN/A value error".format(e))

def field_file(path):
  experiments = os.listdir(log_dir)
  experiments.sort()
  for e in experiments:
    exp_dir = os.path.join(log_dir, e)
    log_file = log_files(exp_dir)
    num_trainers = len(list(log_file)) // 2
    data = []
    title, data = read_itr_log_exp(exp_dir)
    if title is None or data.shape[0] < num_trainers or num_trainers <= 0:
      continue
    try:
      for i in range(len(title)):
        np.savetxt(os.path.join(exp_dir, title[i] + ".txt"), data[:, :, i].T)
    except (ValueError, IndexError):
      print("{} N/A".format(e))

if __name__ == "__main__":
  if len(sys.argv) <= 1:
    dir_name = "latest"
  else:
    dir_name = sys.argv[1]
  log_dir = os.path.join(LOG_PATH, dir_name)
  #read_epoch_time(log_dir)
  components(log_dir)
  #diff_backward_time(log_dir)
  field_file(log_dir)