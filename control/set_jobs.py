import subprocess
import sys
from itertools import product

def name_it(use_permcovar,split,dataset,n_train_classes,lr,n_iters):
    return '_dataset_'+dataset+'_lr_'+lr+'_'+n_train_classes+'_classes_tg_init++_em_'+n_iters+'_iters'

if __name__ == '__main__':
    argv = sys.argv
    fname = 'jobs.txt' if len(argv)<2 else argv[1]
f = open(fname,'wb+')

permcovar_list = ['False']
split_list = ['0']
dataset_list = ['0']
n_train_classes_list = ['5','10','20']
lr_list = ['1e-6']
n_iter_list = ['1','3','5','10']
options = product(permcovar_list,split_list,dataset_list,n_train_classes_list,lr_list,n_iter_list)
for opt in options:
   opt = list(opt)+[name_it(*opt)]
   f.write(str(opt)+'\n')
f.close()
