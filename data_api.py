from inspect import currentframe, getframeinfo
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
logfile_path = project_dir+'/log_test.txt'
def log_print(*msg):
    with open(logfile_path,'a+') as logfile:
        msg = [str(m) for m in msg]
        logfile.write(' '.join(msg))
        logfile.write('\n')

def echo(x):
    print 'loading:',x
    return x

def img_animate(tensor):
    import matplotlib.animation as anim
    import types
    # args:
    #   - tensor: [t,d,d] array
    fig = plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ims=[]
    for time in xrange(np.shape(tensor)[0]):
        im = ax1.imshow(tensor[time,:,:],cmap='gray')
        ims.append([im])
    #run animation
    ani = anim.ArtistAnimation(fig,ims, interval=100,blit=False)
    plt.show()

def binary_display(a,title = ""):
    if type(a)==list:
        # todo: add support for 2< images
        fig = plt.figure()
        sub = fig.add_subplot(1,2,1)
        img = Image.fromarray(a[0]*255)
        plt.imshow(img)
        sub.set_title('Prediction')
        sub = fig.add_subplot(1,2,2)
        img = Image.fromarray(a[1]*255)
        plt.imshow(img)
        sub.set_title('Ground Truth')
    else:
        img = Image.fromarray(a*255)
        img.show(title)

def display(a):
    misc.imshow(a)

def show(a):
	plt.imshow(a)
	plt.show()
	
global_color_coord = 0

"""
def get_gmm_data(n1,n2=None,d=3):
    if not n2:
        n2=n1
    mu = d*np.ones(d)
    x1 = np.random.normal(mu,size=(n1,d))
    x2 = np.random.normal(-mu,size=(n2,d))
    x = np.vstack((x1,x2))
    #np.random.shuffle(x)
    return x
"""
""

def get_gaussians(n,d=2,k=2):
    assert k>=1
    xs = np.zeros((0,d))
    ys = np.zeros((0,k))
    for i in range(k):
        mu = 7*i*np.ones((1, d))
        sample = np.random.normal(mu,size=(n,d))
        xs = np.vstack((xs,sample))
        membership_vec = np.zeros((1,k))
        membership_vec[0,i] = 1
        ys = np.vstack((ys,np.tile(membership_vec,(n,1))))
    return xs,ys

def get_unfaithfull_data(n,r=2):
    '''
    Returns data who's gt clustering doesn't correspond to it's KmeansClustering.
    Need to apply a linear transformation in order to restore faithfulness.
    Data will be comprised of 4 unfaithfull clusters, which in turn correspond to 2 gt clusters.
    i.e:
    A,B: gt clustering labels.
    
        A----B
        |    |
       (r)  (r)
        |    |
        A----B

    args:
        - n: number of points per unfaithfull cluster.
        - r: ratio of unfaithfulness  
    '''
    clsts = [[1,0],[0,1]]
    base_r = 10
    x = np.ndarray((0,2))
    y = np.ndarray((0,2))
    for i in range (4):
        mu = (base_r*(i%2),base_r*r*(int(i<2)))
        x_tmp = np.random.normal(mu,size=(n,2))
        x = np.vstack((x,x_tmp))
        clst = clsts[i%2]
        y_tmp = np.tile(clst,(n,1))
        y = np.vstack((y,y_tmp))
    return x,y

def get_gmm_data(n,k=2):
    x = np.ndarray((0,3))
    for i in range (k):
        mu = np.random.uniform(-1,1,size=(1,3))
        mu = 5*mu # keep 'em seperated
        x_tmp = np.random.normal(mu,size=(n,3))
        x = np.vstack((x,x_tmp))
    return x
""
def scatter_2d(x,indices=None):
    if id(indices)!=id(None): indices = np.array(indices)
    def randrange(n, vmin, vmax):
        '''
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        '''
        return (vmax - vmin)*np.random.rand(n) + vmin
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if id(indices)!=id(None):
        cs = ['r','b','g','black'] # is it necessary? doesn't pyplot cycle through colors anyway?
        for i in range(int(max(indices))+1):
            c = cs[i]
            points = x[indices==i]    
            xs,ys = points[:,0],points[:,1]
            ax.scatter(xs,ys,c=c,marker='o')
    else:
        xs,ys = x[:,0],x[:,1]
        ax.scatter(xs,ys,marker='o')
    plt.show()

def scatter_3d(x,indices=None,title=None):
    if id(indices)!=id(None): indices = np.array(indices)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if id(indices)!=id(None):
        cs = ['r','b','g','k','c','m','y','w'] # todo: add more colors. is it necessary? doesn't pyplot cycle through colors anyway?
        for i in range(int(max(indices))+1):
            c = cs[i]
            points = x[indices==i]    
            xs,ys,zs = points[:,0],points[:,1],points[:,2]
            ax.scatter(xs,ys,zs,c=c,marker='o')
            
    else:
        xs,ys,zs = x[:,0],x[:,1],x[:,2]
        ax.scatter(xs,ys,zs,marker='o')
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z');
    if title!=None: plt.title(title)
    plt.show()

def scatter(x,indices=None):
    d = x.shape[1]
    if d == 2:
        scatter_2d(x,indices)
    elif d == 3:
        scatter_3d(x,indices)
    else:
        print('Could not visualize data. Dimensionality > 3 ...')

def noisify(x):
    shape = x.shape
    #eps = 0.1
    eps = .15
    noise = np.random.normal(0,eps,shape)
    return x+noise

def flip_noisify(arr,flip_ratio=0.2):
    import random
    num_flips = flip_ratio*arr.shape[0]*arr.shape[1]
    num_flips = int(num_flips)
    for flip in range(num_flips):
        i,j = random.randint(0,arr.shape[0]-1),random.randint(0,arr.shape[1]-1)
        arr[i,j] = 1-arr[i,j] # flip
    return arr

def get_relevant_fnames(file_names, class_name):
    ret = []
    for fname in file_names:
        fname = fname.split(' ')[1]
        if fname.split('/')[0]==class_name:
            ret.append(fname)
    return ret

def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)  
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

# train globals
train_data_dirs = ['/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds/CUB_200_2011','/specific/netapp5_2/gamir/carmonda/research/vision/stanford_cars','/specific/netapp5_2/gamir/carmonda/research/vision/stanford_products/permuted_train_data','/specific/netapp5_2/gamir/carmonda/research/vision/oxford_flowers/train']    
train_classes_lst =[range(1,101),range(1,99),range(1,513),range(1,103)] 
szs_lst = [np.array(pickle.load(open(data_dir+'/lengths.pickle'))[:len(train_classes)]) for data_dir,train_classes in zip(train_data_dirs,train_classes_lst)]
offsets_lst = np.array([np.cumsum(szs) for szs in szs_lst])-szs_lst 
train_data = None # store data on RAM for faster access
fixed_ys = None

def get_fixed_ys(n,k):
    assignment_islands = [np.ones((n/k,1)) for i in range(k)]
    ys_assignment = block_diag(*assignment_islands) # assignment matrix
    return ys_assignment

def get_szs_and_offsets(batch_classes,dataset_flag):
    szs = szs_lst[dataset_flag][batch_classes-1]
    offsets = offsets_lst[dataset_flag][batch_classes-1] 
    return szs,offsets

def init_train_data(dataset_flag,mini=False):
    '''
    dataset_flag: which dataset we use
    mini: inidicates whether we use a minisplit or not
    '''
    global train_data,fixed_ys,train_data_dir,train_classes
    #try:
    #    train_data = sa.attach('shm://train_data{}.npy'.format(dataset_flag))
    #except:
    train_data_dir = train_data_dirs[dataset_flag]
    train_classes = train_classes_lst[dataset_flag]
    if mini: # remove half of classes
        train_data = np.load(train_data_dir+'/mini_train_data.npy')
        train_classes = train_classes[:len(train_classes)/2]
    else:
        train_data = np.load(train_data_dir+'/train_data.npy')

def get_train_batch(dataset_flag,k,n,use_crop=False,recompute_ys=False):
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
        print '493:',dataset_flag,k,n,recompute_ys,'train_data.shape:',train_data.shape,'batch_classes:',batch_classes,'clas_szs:',class_szs,'class_offsets:',class_offsets,'img_inds_relative:',img_inds_relative,'final_inds:',final_inds
        exit(0)

    if recompute_ys:
        class_szs = [c.shape[0] for c in loaded_data]
        assignment_islands = [np.ones((sz,1)) for sz in class_szs]
        ys_assignment = block_diag(*assignment_islands) # assignment matrix
    else:
        if fixed_ys is None:
            fixed_ys = get_fixed_ys(n,k)
        ys_assignment = fixed_ys
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
    if mini: class_szs = class_szs[:len(class_szs)/2]
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
    '''
    try:
        xs = np.memmap(xs_name,dtype='float32',mode='r+',shape=shape)
    except:
        print 'creating xs variable'
        xs = np.memmap(xs_name,dtype='float32',mode='w+',shape=shape)
        loaded_data = [np.load(path) if echo(path) else None for path in data_paths]
        print 'finished loading. removing augmentation'
        loaded_data = [c[:len(c)/2] for c in loaded_data] # remove augmentation
        print 'removed augmentation. concatenating'
        xs[...] = np.concatenate(loaded_data)
    '''
    membership_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_membership = block_diag(*membership_islands) # membership matrix
    return xs,ys_membership

def l2_normalize(arr):
    arr_norms = np.sqrt(np.sum(arr**2,1))
    arr_norms = np.reshape(arr_norms,[arr.shape[0],1])
    return arr/arr_norms

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
    log_print('333')
    ddp ='/specific/netapp5_2/gamir/carmonda/research/vision/' # data dir prefix
    data_dirs = [ddp+'caltech_birds/CUB_200_2011',ddp+'stanford_cars',ddp+'stanford_products/permuted_train_data',ddp+'oxford_flowers/total']
    train_inds_list = [range(1,101),range(1,99),range(1,513),range(1,103)]
    test_inds_list = [range(101,201),range(99,197),None,range(103,205)]
    val_inds_list = [None,None,None,range(205,307)]
    minitrain_inds_list = [range(1,51),range(1,99)]
    minitest_inds_list = [range(51,101),range(99,197)]
    split_list = [train_inds_list,test_inds_list,val_inds_list,minitrain_inds_list,minitest_inds_list]
    inds = split_list[split_flag][dataset_flag]
    mini = split_flag>2
    if inds==None: 
        print 'unsupported split:',split_flag,dataset_flag
        print 'for ebay test split, use "restore_and_test_ebay_total.py"'
    data_dir = data_dirs[dataset_flag]
    return load_specific_data(data_dir,inds,mini=mini)
