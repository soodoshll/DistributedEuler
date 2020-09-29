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

    ps = []
    ps_id = 0
    for host in hosts:
        for i in range(args.num_ps):
            ps.append(host + ":" + str(PS_PORT + i))
    worker = []
    worker_id = 0
    for host in hosts:
        for i in range(args.num_workers):
            worker.append(host + ":" + str(WORKER_PORT + i))
    ps_arg = ",".join(ps)
    worker_arg = ",".join(worker)
    command = "cd ~/graph-learn ; source env.sh ; cd ~/gl_distributed; " + udf_command + " --ps_hosts {} --worker_hosts {}".format(ps_arg, worker_arg) 
    print(command)

    # start ps
    ps_id = 0
    for host in hosts:
        for i in range(args.num_ps):
            ps_command = command + " --job_name ps --task_index {} 2> ps{}.err > ps{}.out".format(ps_id, i, i)
            print(ps_command)
            execute_remote(ps_command, host, args.ssh_port, thread_list)
            ps_id += 1
            time.sleep(0.5)
    # start worker
    worker_id = 0
    for host in hosts:
        for i in range(args.num_workers):
            worker_command = command + " --job_name worker --task_index {} 2> worker{}.err > worker{}.out".format(worker_id, i, i)
            print(worker_command)
            execute_remote(worker_command, host, args.ssh_port, thread_list)
            worker_id += 1
            time.sleep(1)

    for thread in thread_list:
        thread.join()

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--ssh_port', type=int, default=22, help='SSH Port.')
    parser.add_argument('--num_workers', type=int, default=1,)
    parser.add_argument('--num_ps', type=int, default=1,)
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
