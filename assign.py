import subprocess
import shlex
import math
from dcdb import *
import os
import sys
import pdb
import datetime
import time
import Queue

# globals
home_dir = '/a/home/cc/cs/carmonda'
project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project'
username = 'carmonda'
class Worker():
    def __init__(self,descriptor):
        tmp = descriptor.split(' ')
        self.gpu = tmp[0]
        self.machine = tmp[1]
        self.job = None
        self.string = 'Worker(machine: {}, gpu: {})'.format(self.machine,str(self.gpu))
        self.nvidia_watcher = None
    def __str__(self):
        return self.string
    def do(self,job):
        self.job = job
        what_to_run = 'restore_and_test.py' if job.test else 'main.py '
        log_name = 'test' if job.test else 'main'

        cmd_prefix = 'ssh {}@{} '.format(username,self.machine)
        cmd_body = 'python {}/{} '.format(project_dir,what_to_run)
        flags = ['--gpu {}'.format(self.gpu)]+['--{} {}'.format(key,job.d[key]) for key in job.d.keys()]
        cmd_flags = ' '.join(flags)
        cmd_piping = ' >>& {}/log_{}.txt'.format(project_dir,log_name)
        cmd = cmd_prefix+cmd_body+cmd_flags+cmd_piping
        
        '----------------'
        ssh_tunnel_cmd = 'python ssh_tunnel.py '
        log_file = 'log_{}.txt'.format('debug_tunnel')
        cmd_at_host = '"'+cmd_body+cmd_flags+cmd_piping+'"'
        new_cmd = ssh_tunnel_cmd+' '.join([username,self.machine,log_file,cmd_at_host])
        self.ps = subprocess.Popen(new_cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
        '----------------'
        watch_cmd = cmd_prefix+'python {}/nvidia_watcher.py {} {} {}'.format(project_dir,self.gpu,self.machine,str(self.ps.pid))
        watch_cmd = shlex.split(watch_cmd) # tokenize
        self.nvidia_watcher = subprocess.Popen(watch_cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    def kill_watch(self):
        nw = self.nvidia_watcher
        try:
            nw.kill()
            nw.communicate()
        except:
            print 'could not kill nw'
            
    def clear(self):
        def clear_ps(ps):
            ps.kill()
            outs, errs = ps.communicate()
        try:
            pid = self.ps.pid
            clear_ps(self.ps)
            clear_ps(self.nvidia_watcher)
            print 'cleared {} with pid {}'.format(self,pid)
        except:
            print "can't clear worker"

class Job():
    def __init__(self,descriptor,train_or_test):
        self.test = train_or_test # 0/1 flag
        parsed = (descriptor[2:-2]).split("', '")
        self.string = str(parsed)
        self.d = dict()
        self.d['deepset'],self.d['data_split'],self.d['dataset'],self.d['n_train_classes'],self.d['model_lr'],self.d['n_iters'],self.d['name'] = parsed
    def __str__(self):
        return 'Job('+self.string+' '+str(self.test)+')'

class Watcher(): # collates and displays different nvidia-smi queries
    def __init__(self,active_workers):
        self.watch_dir = project_dir+'/nvidia_watch' # location of nvidia-smi query results
        self.update_workers(active_workers)
        self.finished_jobs = []
    def update_workers(self,active_workers): # update which processes to collate queries for
        self.active_workers = active_workers
        self.pids = [worker.ps.pid for worker in active_workers]
        self.saw_pids = [False]*len(self.pids)
        self.job_lines = ['']*2*len(self.pids) # initialize query result
        m = len(active_workers)
        list_aw = list(active_workers)
        self.assign_msgs = []
        for i in range(m):
            worker = list_aw[i]
            job = worker.job
            assign_msg = '{}: assigned {} to {}'.format(str(i),job,worker)
            self.assign_msgs.append(assign_msg)
    def add_finished(self,job):
        self.finished_jobs.append(job)
    def collate(self):
        now = str(datetime.datetime.now())
        header_lines = [now,'+-----------------------------------------------------------------------------+', '| NVIDIA-SMI 384.98                 Driver Version: 384.98                    |', '|-------------------------------+----------------------+----------------------+', '| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |', '| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |', '|===============================+======================+======================|']
        header_lines = ['GPU Usage:','----------']+[' '*20+line for line in header_lines] # tilt header
        # update relevant line fields:
        for i,pid in indexise(self.pids):
            filename = self.watch_dir+'/'+str(pid)+'_watch.txt'
            try:
                with open(filename,'r') as f:
                    content = f.readlines()
                    self.job_lines[2*i] = content[0]
                    self.job_lines[2*i+1] = content[1]
                    self.saw_pids[i] = True
            except:
                pass
        delimiter = ' '*20+'+'+'-'*77+'+'
        self.cls()
        print 'Active Jobs:'
        print '------------'
        for msg in self.assign_msgs:
            print msg
        print ''
        for line in header_lines:
            print line.rstrip()
        for i,line in indexise(self.job_lines):
            pid_ind = int(math.floor(i/2))
            if self.saw_pids[pid_ind]:
                print line.rstrip()
        print delimiter
        print 'Finished Jobs:'
        print '--------------'
        for job in self.finished_jobs: print job

    def cls(self):
        os.system('clear')

    @staticmethod
    def rm(pid):
        watch_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project/nvidia_watch' # hard coded for safety
        fname = watch_dir+'/'+str(pid)+'_watch.txt'
        os.system('rm {}'.format(fname))


def indexise(lst):
    return zip(range(len(lst)),lst)

def enqueue_list(l):
    q = Queue.Queue()
    for e in l:
        q.put(e)
    return q

if __name__ == "__main__":
    os.system('get_workers.sh') # run script that checks which gpus are available
    argv = sys.argv
    train_or_test = bool(int(argv[2]))
    #watch_fname = 'watch_nvidia_'+argv[1].split('.')[0]+'_'+int(argv[2])*'test'+(1-int(argv[2]))*'train'+'.txt' # name of status file
    path_to_jobs = argv[1]
    path_to_workers = home_dir+'/workers.txt'
    f_jobs = open(path_to_jobs,'r')
    f_workers = open(path_to_workers,'r')
    job_descriptors = f_jobs.readlines()
    worker_descriptors = f_workers.readlines()
    f_jobs.close()
    f_workers.close()
    jobs = [Job(l[:-1],train_or_test) for l in job_descriptors]
    job_queue = enqueue_list(jobs)
    workers = [Worker(l[:-1]) for l in worker_descriptors]
    m = min(len(jobs),len(workers))
    for i in range(m):
        worker,job = workers[i],job_queue.get()
        worker.do(job)
    active_workers = set(workers[:m])
    if m==0:
        print 'no available workers. try later or relax constraints'
        exit()
    try:
        watcher = Watcher(active_workers)
        while len(active_workers)>0:
            watcher.update_workers(active_workers)
            watcher.collate()
            time.sleep(1)
            idle_workers = set([])
            for worker in active_workers:
                response = worker.ps.poll() # remove zombie
                if response != None: 
                    #print '{} finished {} with response {}'.format(worker,worker.job,response)
                    watcher.add_finished(worker.job)
                    Watcher.rm(worker.ps.pid)
                    worker.kill_watch()
                    if not job_queue.empty(): # get next job from queue
                        #print 'queue not empty'
                        job = job_queue.get()
                        worker.do(job)
                    else: 
                        idle_workers.add(worker)
            active_workers = active_workers.difference(idle_workers)
        watcher.update_workers(active_workers)
        watcher.collate() # print all finished report
        print 'finished!'
    except: # catches ctrl+c signal
        print 'error occured. exiting work loop'
        for worker in active_workers: 
            worker.clear()
            Watcher.rm(worker.ps.pid)
    print 'exiting program'
