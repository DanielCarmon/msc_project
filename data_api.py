from inspect import currentframe, getframeinfo
import os
#import SharedArray
import pickle
import signal
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
from PIL import Image
from scipy.linalg import block_diag
import pdb
from scipy.misc import imread, imresize
rand = np.random.randint

project_dir = "/specific/netapp5_2/gamir/carmonda/research/vision/msc_project"
logfile_path = project_dir+'/log_data_api.txt'
def log_print(*msg):
    with open(logfile_path,'a+') as logfile:
        msg = [str(m) for m in msg]
        logfile.write(' '.join(msg))
        logfile.write('\n')

def echo(x):
    print 'loading:',x
    return x

# train globals
train_data_dirs = ['/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds/CUB_200_2011',
                   '/specific/netapp5_2/gamir/carmonda/research/vision/stanford_cars',
                   '/specific/netapp5_2/gamir/carmonda/research/vision/stanford_products/permuted_train_data',
                   '/specific/netapp5_2/gamir/carmonda/research/vision/oxford_flowers/train'
                   ]    
train_classes_lst =[range(1,101),range(1,99),range(1,513),range(1,103)] 
szs_lst = [np.array(pickle.load(open(data_dir_+'/lengths.pickle'))[:len(train_classes_)]) for data_dir_,train_classes_ in zip(train_data_dirs,train_classes_lst)]
offsets_lst = np.array([np.cumsum(szs) for szs in szs_lst])-szs_lst 
train_data = None # store data on RAM for faster access
fixed_ys = None

def get_fixed_ys(n,k):
    assignment_islands = [np.ones((n/k,1)) for i in range(k)]
    ys_assignment = block_diag(*assignment_islands) # assignment matrix
    return ys_assignment

def get_szs_and_offsets(batch_classes,dataset_flag):
    global szs_lst,offsets_lst
    szs = szs_lst[dataset_flag][batch_classes-1]
    offsets = offsets_lst[dataset_flag][batch_classes-1] 
    return szs,offsets

def refresh_train_data_and_ls(dataset_flag,current_batch,mini=False,name=''):
    global train_data,train_data_dirs,szs_lst,offsets_lst
    train_data_dirs = ['/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds/CUB_200_2011',
                       '/specific/netapp5_2/gamir/carmonda/research/vision/stanford_cars',
                       '/specific/netapp5_2/gamir/carmonda/research/vision/stanford_products/permuted_train_data',
                       '/specific/netapp5_2/gamir/carmonda/research/vision/oxford_flowers/train'
                       ]   # reset 
    train_data_dir = train_data_dirs[dataset_flag] 
    szs_lst = [np.array(pickle.load(open(data_dir+'/lengths.pickle'))[:len(train_classes)]) for data_dir,train_classes in zip(train_data_dirs,train_classes_lst)] # get lengths file based on updated paths
    offsets_lst = np.array([np.cumsum(szs) for szs in szs_lst])-szs_lst # calculate offsets 
    train_data = np.load(train_data_dir+'/train_data.npy') # get new data

def init_train_data(dataset_flag,mini=False,name=''):
    '''
    dataset_flag: which dataset we use
    mini: inidicates whether we use a minisplit or not
    '''
    global train_data,fixed_ys,train_data_dir,train_classes
    train_data_dir = train_data_dirs[dataset_flag]
    train_classes = train_classes_lst[dataset_flag]
    pdb.set_trace()
    if mini: # remove half of classes
        train_data = np.load(train_data_dir+'/mini_train_data.npy')
        train_classes = train_classes[:len(train_classes)/2]
    else:
        try:
            train_data = np.load(train_data_dir+'/train_data.npy')
        except:
            log_print('exception when loading train data')
            pdb.set_trace()

def get_train_batch(dataset_flag,k,n,use_crop=False,recompute_ys=False,name=''):
    global fixed_ys
    n_per_class = int(n/k)
    
    # sample
    batch_classes = np.random.choice(train_classes,k,replace=False)
    class_szs,class_offsets = get_szs_and_offsets(batch_classes,dataset_flag)
    img_inds_relative = [np.random.choice(class_sz,min(n_per_class,class_sz),replace=False) for class_sz in class_szs] # intra-class indices of sampled imgs.
    final_inds = np.concatenate([class_offset+img_inds for class_offset,img_inds in zip(class_offsets,img_inds_relative)])
    # slice
    try:
        xs = train_data[final_inds]
    except:
        exc =  sys.exc_info()
        traceback.print_exception(*exc)
        print 'exception for:',name 
        log_print('excpetion with train data load')
        pdb.set_trace()

    if recompute_ys:
        class_szs = [len(x) for x in img_inds_relative]
        assignment_islands = [np.ones((sz,1)) for sz in class_szs]
        ys_assignment = block_diag(*assignment_islands) # assignment matrix
    else:
        if fixed_ys is None:
            fixed_ys = get_fixed_ys(n,k)
        ys_assignment = fixed_ys
    #np.save('test_me',xs)
    #np.save('test_me_ls',class_szs)
    return xs,ys_assignment

def get_len_list(inds,data_dir,augment):
    lengths = pickle.load(open(data_dir+'/lengths.pickle')) # file with number of imgs in each class. each data_dir needs to contain this file.
    ret = []
    for i in inds:
        to_append = lengths[i-1]
        if not augment: to_append = to_append/2 # don't count flipped images
        ret.append(to_append)
    return ret

def load_specific_data(data_dir,inds,augment=False,use_crop=False,mini=False):
    version = ''
    if use_crop: version = '_cropped'
    data_paths = [data_dir+"/class"+str(i)+"{}.npy".format(version) for i in inds]
    class_szs = get_len_list(inds,data_dir,augment)
    shape = sum(class_szs),299,299,3
    which_data = str(inds[0])+"_to_"+str(inds[-1])
    which_dataset = data_dir.split('/')[-1]
    xs_name = which_dataset+'_'+which_data+'_xs{}'.format(version)
    xs_name = data_dir+'/'+xs_name
    if augment: xs_name+='_augmented'
    log_print('reading xs from {}...'.format(xs_name))
    try:
        xs = np.memmap(xs_name,dtype='float32',mode='r+',shape=shape)
    except:
        log_print('failed to read {}. exiting program'.format(xs_name))
        exit(0)
    log_print('read xs with shape {}'.format(xs.shape))
    membership_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_membership = block_diag(*membership_islands) # membership matrix
    return xs,ys_membership

def get_data(split_flag,dataset_flag):
    '''
    split_flag:
        0: train
        1: test
        2: val
        3: minitrain
        4: minitest
    dataset_flag:
        0: birds
        1: cars
        2: products
        3: flowers
    '''
    log_print('in get_data function')
    ddp ='/specific/netapp5_2/gamir/carmonda/research/vision/' # data dir prefix
    data_dirs = [ddp+'caltech_birds/CUB_200_2011',ddp+'stanford_cars',ddp+'stanford_products/permuted_train_data',ddp+'oxford_flowers/total']
    train_inds_list = [range(1,101),range(1,99),range(1,513),range(1,103)]
    test_inds_list = [range(101,201),range(99,197),None,range(103,205)]
    val_inds_list = [None,None,range(512),range(205,307)]
    minitrain_inds_list = [range(1,51),range(1,50)] 
    minitest_inds_list = [range(51,101),range(50,99)]
    split_list = [train_inds_list,test_inds_list,val_inds_list,minitrain_inds_list,minitest_inds_list]
    inds = split_list[split_flag][dataset_flag]
    mini = split_flag>2
    if inds==None: 
        print 'unsupported split:',split_flag,dataset_flag
        print 'for ebay test split, use "restore_and_test_ebay_total.py"'
    data_dir = data_dirs[dataset_flag]
    return load_specific_data(data_dir,inds,mini=mini)
