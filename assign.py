import subprocess
import os
import sys
import pdb
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
    def __str__(self):
        return self.string
    def do(self,job):
        self.job = job
        what_to_run = 'restore_and_test.py' if job.test else 'main.py 4'
        log_name = 'test' if job.test else 'main'

        cmd_prefix = 'ssh {}@{} '.format(username,self.machine)
        cmd_body = 'python {}/{} '.format(project_dir,what_to_run)
        flags = ['--gpu {}'.format(self.gpu)]+['--{} {}'.format(key,job.d[key]) for key in job.d.keys()]
        cmd_suffix = ' '.join(flags)
        cmd = cmd_prefix+cmd_body+cmd_suffix
        cmd += ' 2>> ~/log_{}.txt'.format(log_name)
        self.ps = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
    def terminate(self):
        try:
            pid = self.ps.pid
            self.ps.kill()
            print 'terminated worker {} with pid {}'.format(self,pid)
        except:
            print "can't terminate idle worker"

class Job():
    def __init__(self,descriptor,train_or_test):
        self.test = train_or_test # 0/1 flag
        parsed = (descriptor[2:-2]).split("', '")
        self.string = str(parsed)
        self.d = dict()
        self.d['deepset'],self.d['data_split'],self.d['dataset'],self.d['n_train_classes'],self.d['model_lr'],self.d['n_iters'],self.d['name'] = parsed
    def __str__(self):
        return 'Job('+self.string+' '+str(self.test)+')'

def enqueue_list(l):
    q = Queue.Queue()
    for e in l:
        q.put(e)
    return q

if __name__ == "__main__":
    os.system('get_workers.sh') # run script that checks which gpus are available
    #time.sleep(3) # wait 5 secs
    argv = sys.argv
    train_or_test = bool(int(argv[2]))
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
        print '{}: assigning job {} to worker {}'.format(str(i),job,worker)
        worker.do(job)
    active_workers = set(workers[:m])
    if m==0:
        print 'no available workers. try later or relax constraints'
        exit()
    print 'working...'
    while len(active_workers)>0:
        time.sleep(3)
        idle_workers = []
        for worker in active_workers:
            response = worker.ps.poll() # remove zombie
            if response != None: 
                print 'worker {} finished job: {}'.format(worker,worker.job)
                if not job_queue.empty(): # get next job from queue
                    job = job_queue.get()
                    print 'assigning job {} to worker {}'.format(job,worker)
                    worker.do(job)
                else:
                    idle_workers.append(worker)
        for worker in idle_workers:
            active_workers.discard(worker) # don't poll again if finished or not
    print 'finished!'
