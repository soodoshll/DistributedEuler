"""Launching tool for DGL distributed training"""
import os
import stat
import sys
import subprocess
import argparse
import signal
import time
import json
from threading import Thread

PS_PORT = 2222
CHIEF_PORT = 3333
WORKER_PORT = 4444

def execute_remote(cmd, ip, port, thread_list):
    """execute command line on remote machine via ssh"""
    cmd = 'ssh -p ' + str(port) + ' ' + ip + ' \'' + cmd + '\''
    print(cmd)
    # thread func to run the job
    def run(cmd):
        subprocess.run(cmd, shell = True)

    thread = Thread(target = run, args=(cmd,))
    thread.setDaemon(True)
    thread.start()
    thread_list.append(thread)

def submit_jobs(args, udf_command):
    """Submit distributed jobs (server and client processes) via ssh"""
    # hosts = []
    thread_list = []
    
    with open(args.ip_config) as f:
        hosts = f.readlines()
        hosts = [host.strip() for host in hosts]

    ps = [hosts[0]+":"+str(PS_PORT)]
    chief = [hosts[0]+":"+str(CHIEF_PORT)]
    worker = []
    for host in hosts:
        for i in range(args.num_workers):
            if i == 0 and host == hosts[0]:
                continue
            worker.append(host + ":" + str(WORKER_PORT + i))
    ps_arg = ",".join(ps)
    chief_arg = ",".join(chief)
    worker_arg = ",".join(worker)
    command =udf_command + " --ps {} --chief {} --worker {}".format(ps_arg, chief_arg, worker_arg) 
    print(command)

    # start ps
    ps_command =  "cd ~/euler_profile ; " + command + " --task_type ps --task_id 0"
    print(ps_command)
    execute_remote(ps_command, hosts[0], args.ssh_port, thread_list)
    # start cheif
    threads_per_worker = 48 // args.num_workers
    omp_num_threads=1
    command = "TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 "  + command
    chief_command = "cd ~/euler_profile ; " + command + " --task_type chief --task_id 0" 
    print(chief_command)
    execute_remote(chief_command, hosts[0], args.ssh_port, thread_list)
    # start worker
    worker_id = 0
    for host in hosts:
        for i in range(args.num_workers):
            if i == 0 and host == hosts[0]:
                continue
            worker_command =  "cd ~/euler_profile ; TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 " + command + " --task_type worker --task_id {} 2> worker{}.err > worker{}.out".format(worker_id, worker_id, worker_id)
            worker_id += 1
            print(worker_command)
            execute_remote(worker_command, host, args.ssh_port, thread_list)


    # execute_remote("ls", 
    for thread in thread_list:
        thread.join()

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--ssh_port', type=int, default=22, help='SSH Port.')
    parser.add_argument('--num_workers', type=int, default=1,)
    # parser.add_argument('--num_ps', type=int, default=1,)
    parser.add_argument('--ip_config', type=str, default="ip_config",
                        help='The file (in workspace) of IP configuration for server processes')
    args, udf_command = parser.parse_known_args()
    assert len(udf_command) == 1, 'Please provide user command line.'
    udf_command = str(udf_command[0])
    submit_jobs(args, udf_command)

def signal_handler(signal, frame):
    sys.exit(0)

if __name__ == '__main__':
    # fmt = '%(asctime)s %(levelname)s %(message)s'
    # logging.basicConfig(format=fmt, level=logging.INFO)
    # signal.signal(signal.SIGINT, signal_handler)
    main()
