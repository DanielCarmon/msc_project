import os
import pdb
import subprocess
import datetime
import sys
import time

project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project'
argv = sys.argv
gpu_ind = int(argv[1])
machine = argv[2]
pid = argv[3]
spid = str(os.getpid())
self_desc = 'watcher {} {} {}. spid = {}'.format(argv[1],argv[2],argv[3],spid)
f_path = project_dir+'/nvidia_watch/{}_watch.txt'.format(pid)
with open(f_path,'w+') as f:
    #f.write(self_desc+'\n')
    #pdb.set_trace()
    while os.path.exists(f_path):    
        lines = subprocess.check_output(['nvidia-smi']).split('\n')
        gpu_ind_to_linenum = lambda i: 7+3*i # number of relevant line in nvidia-smi output
        linenum = gpu_ind_to_linenum(gpu_ind)
        
        string1 = 'PID {}, GPU {}:'.format(pid,str(gpu_ind))
        string2 =  string1+' '*(20-len(string1))+lines[linenum]
        string3 =  '({})'.format(machine)+' '*(20-2-len(machine))+lines[linenum+1]
        f.write(string2+'\n')
        f.write(string3+'\n')
        f.flush()
        f.seek(0) # overwrite prev stat
        time.sleep(0.05) # avoid screen flickering
    #msg = 'exception occured in '+self_desc 
    #os.system("echo '{}' >> new_log_file.txt".format(msg))
    os.system('kill -9 {}'.format(spid))
