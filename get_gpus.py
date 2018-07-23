import subprocess
import pdb
from dcdb import cp

machine = subprocess.check_output("echo $HOST",stderr=subprocess.STDOUT,shell=True)
column = subprocess.check_output("nvidia-smi | awk '{print $2}'",stderr=subprocess.STDOUT,shell=True)
used = set([x for x in (column.split('GPU')[-1]).split('\n') if x!=''][1:])
total = set([str(i) for i in range(4)])
free = list(total.difference(used))
f = open('workers.txt','a')
if (machine[:1]=='r'): # don't overload rack machines
    margin = 2
    current_load = len(used)
    margin_load = margin-current_load
    free = free[:max(0,margin_load)]
for gpu in free:
    f.write(gpu+' '+machine)
f.close()
