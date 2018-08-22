import subprocess
import os
import pdb
machine = subprocess.check_output("echo $HOST",stderr=subprocess.STDOUT,shell=True)
column = subprocess.check_output("nvidia-smi | awk '{print $3}'",stderr=subprocess.STDOUT,shell=True)
pids = set([x for x in (column.split('PID')[-1]).split('\n') if x!=''])
for pid in pids:
    os.system('kill -9 {}'.format(pid))
