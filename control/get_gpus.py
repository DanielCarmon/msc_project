import subprocess
import os
import pdb
from dcdb import cp

#su0_list = ['pc-wolf-g01','pc-wolf-g02','pc-gamir-g01','rack-gamir-g01','rack-gamir-g02'] # machines on which system uses gpu 0
su0_list = [] # machines on which system uses gpu 0
machine = subprocess.check_output("echo $HOST",stderr=subprocess.STDOUT,shell=True)
su0 = machine[:-1] in su0_list
column = subprocess.check_output("nvidia-smi | awk '{print $2}'",stderr=subprocess.STDOUT,shell=True)
if su0:
    used = set([x for x in (column.split('GPU')[-1]).split('\n') if x!=''][1:])
else:
    used = set([x for x in (column.split('GPU')[-1]).split('\n') if x!=''])
if machine=='savant\n':
    total = set([str(i) for i in range(6)])
else:
    total = set([str(i) for i in range(4)])
free = list(total.difference(used))
f = open('workers.txt','a')
if ('gamir' in machine and 'rack' in machine): # don't overload rack-gamir machines
    margin = 2
    current_load = len(used)
    margin_load = margin-current_load
    free = free[:max(0,margin_load)]
for gpu in free:
    f.write(gpu+' '+machine)
f.close()
