import pickle
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
global_color_coord = 0
def add_object(x,D='rects'):
    global global_color_coord
    if D=='rects':
        # adds a colored rectangle to p
        d = x.shape[0]
        corner1 = rand(0,d,2)
        corner2 = rand(0,d,2)
        miny,maxy,minx,maxx = min(corner1[1],corner2[1]),max(corner1[1],corner2[1]),min(corner1[0],corner2[0]),max(corner1[0],corner2[0])
        #c = rand(0,3) # color
        x[minx:maxx,miny:maxy] = 0
        x[minx:maxx,miny:maxy,global_color_coord] = 1
        obj = np.zeros((d,d,3))
        obj[minx:maxx,miny:maxy,global_color_coord] = 1
        #c+=1
        #c%=3
        return obj
    else:
        return NotImplemented
def get_background_mask(y):
    bg = np.ones(y[0].shape[:2])
    for mask in y:
        bg*=1-np.max(mask,2)
    return bg
def combine_masks(y):
    # args:
    #   - y: list of dxd binary mask matrices
    # output:
    #   - ret: (d^2)x(d^2) binary is-same-cluster matrix
    d = len(y[0])
    ret = np.ones((d**2,d**2))
    latest = np.ones((d,d))
    #mask_log,tmp_log,mask_vec_log = [],[],[]
    i = 0
    #pdb.set_trace()
    for mask in y[::-1]:
        tmp = np.sum(mask,axis=2)
        tmp *= latest # remove previously seen pixels
        mask_vec = np.reshape(tmp,(d**2,1))
        #mask_log.append(mask),tmp_log.append(tmp),mask_vec_log.append(mask_vec)
        i+=1
        ret *= 1-mask_vec*mask_vec.T
        latest *= 1-tmp
    print ret
    return 1-ret
def get_img(n,k=1,d=28,exact_count=True):
    """
    Returns n images made of rgb squares
    label: segmentation mask
    k = avg num of obj. if exact_count then k == num objs
    """
    global global_color_coord
    xs = np.zeros((n,d,d,3))
    ys = []
    for x in xs:
        masks = np.zeros((d,d)) # book-keeping variable to count num of visible objs
        y = []
        num_objects = k # for meanwhile
        obj_ind=1
        i=1
        while obj_ind<num_objects+1:
            #print '----- loop iter ------'
            global_color_coord+=1
            global_color_coord%=3
            #print 'adding object of color:',global_color_coord
            new_mask = add_object(x)
            y.append(new_mask)
            #print(np.unique(masks))

            # book-keeping
            new_mask_flat = np.sum(new_mask,axis=2)
            masks *= (1-new_mask_flat) 
            new = (i)*new_mask_flat
            masks+=new
            
            old = obj_ind
            obj_ind = len(np.unique(masks))
            if not old<obj_ind:
                #print 'nothing new'
                global_color_coord-=1
                global_color_coord%=3
            i+=1
        #y = combine_masks(y)
        ys.append(y)
    return xs,ys
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
def get_clst_mat_frommasks(y):
    # returns clst_mat, including background
    # input: list of cluster masks
    d = y[0].shape[0]
    ret = np.zeros((d**2,d**2))
    for mask in y:
        # mask is [d,d,3]
        mask_ = np.sum(mask,axis=2)
        vec = np.reshape(mask_,(d**2,1))
        rel = np.matmul(vec,vec.T) 
        ret*=(1-rel)
        ret+=rel    
    bg_vec = np.array([int(b) for b in ~ret.any(axis=1)]) # loop of len d
    bg_vec = np.reshape(bg_vec,(len(bg_vec),1))
    ret += np.matmul(bg_vec,bg_vec.T) 
    ''' # vectorized approach
    bg_pixels = np.where(~ret.any(axis=1))
    bg_n = len(bg_pixels)
    fill = np.ones((bg_n,bg_n))
    ret[] = fill # substitute submatrix
    '''
    return ret
def get_clst_mat_from1hot(y):
    return np.matmul(y,y.T)
def get_clst_mat(y,flag):
    if flag=='one-hot':
        return get_clst_mat_from1hot(y)
    elif flag=='mask':
        return get_clst_mat_frommasks(y)
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
def preprocess_data(data_dir, save_dir,d,num_classes=200):
    # needs to be called only once.
    # data_dir: where data is currently located
    # save_dir: where preprocessed data should be saved
    # d: size to reshape images to. [d,d,3]
    # num_classes = total number of classes in dataset
    import Image
    sess = tf.InteractiveSession()
    # two lines computation graph:
    data_ph = tf.placeholder(tf.uint8,[None,d,d,3])
    xs_normalized = tf.image.convert_image_dtype(data_ph,tf.float32) # normalize image
    
    xs = np.zeros((0, d, d, 3))
    ys_membership = np.zeros((0, num_classes))
    # fill xs:
    print 'filling xs'
    crop = True
    class_names = open(data_dir + '/classes.txt').readlines()
    file_names = open(data_dir + '/images.txt').readlines()
    curr_class = 0
    for class_name in class_names:
        class_name = class_name.split(' ')[1][:-1]
        xs = np.zeros((0, d, d, 3))
        print class_name
        file_names_relevant = get_relevant_fnames(file_names, class_name)
        for fname in file_names_relevant:
            fname = fname[:-1]
            fclass = fname.split('/')[0]
            if fclass == class_name:
                fname_ = data_dir + '/images/' + fname
                im = Image.open(fname_)
                #pdb.set_trace()
                img_arr = np.array(im)
                if len(img_arr.shape)!=3:
                    print 'bw image!'
                    continue
                if crop: # cropping as described at Zemel et al. Added on 28.3.18
                    print 'cropping',fname
                    #pdb.set_trace() # TODO: SOLVE 0 NMI BUG!
                    img_arr = imresize(img_arr, (256, 256))
                    img_arr = crop_center(img_arr,227,227)
                img_arr = imresize(img_arr, (d, d))  # resize.
                xs = np.vstack((xs, img_arr[np.newaxis, :, :, :]))
        #membership_vec = np.zeros((1, num_classes))
        #membership_vec[0, curr_class] = 1
        #ys_membership = np.vstack((ys_membership, membership_vec))
        curr_class += 1
        feed_dict={data_ph:xs}
        xs = sess.run(xs_normalized,feed_dict=feed_dict)
        xs_flipped = np.flip(xs,2) # horizontal flipping
        xs_final = np.concatenate([xs,xs_flipped])

        '''
        bad preprocess:
        xs_mean = np.mean(xs,0)
        xs_var = np.mean((xs-xs_mean)**2, 0)
        xs_normalized = (xs-xs_mean)/np.sqrt(xs_var)
        '''
        # save xs
        version_name = '' 
        #if crop:
        #    version_name = '_cropped'
        save_dir_ = save_dir + '/class'+str(curr_class)+'{}.npy'.format(version_name)
        print 'saving at',save_dir_
        np.save(save_dir_,xs_final)
    return 0
#data_dir = save_dir = '/home/gamir/carmonda/research/vision/caltech_birds/CUB_200_2011'
#preprocess_data(data_dir,save_dir,299)
def get_bird_train_data(k,n,d):
    '''
    :param k: num classes
    :param n: num data points per class
    :param d: img dims to crop to
    :return: [k*n,d,d,3] , [k*n,k*n]
    '''
    import Image
    xs = []
    ys_membership = np.zeros((0, k))

    data_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds/CUB_200_2011'
    #data_dir = '/home/d/Desktop/uni/research/CUB_200_2011'
    class_names = open(data_dir+'/classes.txt').readlines()
    file_names = open(data_dir+'/images.txt').readlines()
    curr_class = 0
    for class_name in class_names[:k]:
        class_name = class_name.split(' ')[1][:-1]
        file_names_relevant = get_relevant_fnames(file_names,class_name)
        for fname in np.random.permutation(file_names_relevant)[:n]:
            fname = fname[:-1]
            fclass = fname.split('/')[0]
            if fclass==class_name:
                fname_ = data_dir+'/images/'+fname
                im = Image.open(fname_)
                img_arr = np.array(im)
                img_arr = imresize(img_arr, (d, d))  # crop/resize.
                xs.append(img_arr)
                membership_vec = np.zeros((1, k))
                membership_vec[0,curr_class] = 1
                ys_membership = np.vstack((ys_membership, membership_vec))
        curr_class+=1
    ys = np.matmul(ys_membership, ys_membership.T)
    return np.array(xs), ys
def get_bird_test_data(k,n,d):
    '''
    :param k: num classes
    :param n: num data points per class
    :param d: img dims to crop to
    :return: [k*n,d,d,3] , [k*n,k]
    '''
    import Image
    xs = []
    ys_membership = np.zeros((0, k))

    data_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds/CUB_200_2011'
    #data_dir = '/home/d/Desktop/uni/research/CUB_200_2011'
    class_names = open(data_dir+'/classes.txt').readlines()
    file_names = open(data_dir+'/images.txt').readlines()
    curr_class = 0
    for class_name in class_names[-k:]:
        class_name = class_name.split(' ')[1][:-1]
        file_names_relevant = get_relevant_fnames(file_names,class_name)
        for fname in np.random.permutation(file_names_relevant)[:n]:
            fname = fname[:-1]
            fclass = fname.split('/')[0]
            if fclass==class_name:
                fname_ = data_dir+'/images/'+fname
                im = Image.open(fname_)
                img_arr = np.array(im)
                img_arr = imresize(img_arr, (d, d))  # crop/resize.
                xs.append(img_arr)
                membership_vec = np.zeros((1, k))
                membership_vec[0,curr_class] = 1
                ys_membership = np.vstack((ys_membership, membership_vec))
        curr_class+=1
    ys = np.matmul(ys_membership, ys_membership.T)
    return np.array(xs), ys

CUB_loaded_train_data = None # global. data on RAM
cars169_loaded_train_data = None # global. data on RAM
products_loaded_train_data = None # global. data on RAM
loaded_train_data_list = [CUB_loaded_train_data,cars169_loaded_train_data,products_loaded_train_data]
def get_bird_train_data2(data_dir,k,n,n_seen_classes=100,use_crop = False):
    global CUB_loaded_train_data
    train_classes = range(1,1+n_seen_classes) 
    perm = np.random.permutation(train_classes)
    classes = perm[range(k)]
    #print 'classes:',classes
    if CUB_loaded_train_data is None:
        print 'loading train data'
        version= ''
        if use_crop: version = '_cropped' # decide which data augmentation to use. crop+resize or just resize
        # loaded_train_data = [np.load(data_dir+"/class"+str(i)+".npy") for i in train_classes]
        CUB_loaded_train_data = [np.load(data_dir+"/class"+str(i)+"{}.npy".format(version)) for i in range(1,101)]
    loaded_data = [CUB_loaded_train_data[c-1] for c in classes]
    loaded_data = [np.random.permutation(class_data)[:n] for class_data in loaded_data] # take random subsample 
    class_szs = [class_data.shape[0] for class_data in loaded_data]
    assignment_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_assignment = block_diag(*assignment_islands) # assignment matrix
    xs = np.concatenate(loaded_data,0)
    return xs,ys_assignment

def get_train_batch(dataset_flag,k,n,use_crop=False):
    global loaded_train_data_list
    '''
    dataset_flag:
        0: CUB Birds
        1: Stanford Cars
        2: Stanford Products
    k: num of classes in mini-batch
    n: num of datapoints in mini-batch
    '''
    data_dirs = ['/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds/CUB_200_2011','/specific/netapp5_2/gamir/carmonda/research/vision/stanford_cars','/specific/netapp5_2/gamir/carmonda/research/vision/stanford_products/permuted_train_data']    
    data_dir = data_dirs[dataset_flag]
    n_per_class = int(n/k)
    if dataset_flag==0:
        return get_bird_train_data2(data_dir,k,n_per_class)
    train_classes_list = [range(1,101),range(1,99),range(1,513)]
    train_classes = train_classes_list[dataset_flag]
    perm = np.random.permutation(train_classes)
    classes = perm[range(k)]
    if loaded_train_data_list[dataset_flag] is None:
        print 'loading train data'
        version= ''
        if use_crop: version = '_cropped' # decide which data augmentation to use. crop+resize or just resize
        # loaded_train_data_list[dataset_flag] = [np.load(data_dir+"/class"+str(i)+".npy") for i in train_classes]
        loaded_train_data_list[dataset_flag] = [np.load(data_dir+"/class"+str(i)+"{}.npy".format(version)) for i in train_classes]
    loaded_data = [loaded_train_data_list[dataset_flag][c-1] for c in classes]
    loaded_data = [np.random.permutation(class_data)[:n_per_class] for class_data in loaded_data] # take random subsample 
    class_szs = [class_data.shape[0] for class_data in loaded_data]
    assignment_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_assignment = block_diag(*assignment_islands) # assignment matrix
    xs = np.concatenate(loaded_data,0)
    return xs,ys_assignment

def augment(data_dir,version=''):
    """ this should only be called once """
    for i in range(1,201):
        print 'augmenting',i
        class_data_path = data_dir+"/class{}{}.npy".format(str(i),version)
        class_data = np.load(class_data_path)
        class_data_flipped = np.flip(class_data,2) # horizontal flipping
        class_data = np.vstack((class_data,class_data_flipped))
        np.save(class_data_path,class_data)
#augment('/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds',version='_cropped')

def get_len_list(inds,data_dir,augment):
    lengths = pickle.load(open(data_dir+'/lengths.pickle')) # file with number of imgs in each class. each data_dir needs to contain this file.
    ret = []
    for i in inds:
        to_append = lengths[i-1]
        if not augment: to_append = to_append/2 # don't count flipped images
        ret.append(to_append)
    return ret

def load_specific_data(data_dir,inds,augment=False,use_crop=False):
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
    agreement_islands = [np.ones((sz,sz)) for sz in class_szs]
    ys = block_diag(*agreement_islands) # partition matrix
    membership_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_membership = block_diag(*membership_islands) # membership matrix
    return xs,ys,ys_membership

def l2_normalize(arr):
    arr_norms = np.sqrt(np.sum(arr**2,1))
    arr_norms = np.reshape(arr_norms,[arr.shape[0],1])
    return arr/arr_norms

def get_data(split_flag,dataset_flag):
    '''
    split_flag:
        0: train
        1: test
    dataset_flag:
        0: birds
        1: cars
        2: products
    '''
    ddp ='/specific/netapp5_2/gamir/carmonda/research/vision/' # data dir prefix
    data_dirs = [ddp+'caltech_birds/CUB_200_2011',ddp+'stanford_cars',ddp+'stanford_products']
    train_inds_list = [range(1,101),range(1,99),range(1,11319)]
    test_inds_list = [range(101,201),range(99,196),range(11319,22635)]
    split_flag = int(split_flag)
    split_list = [train_inds_list,test_inds_list]
    inds = split_list[split_flag][dataset_flag]
    data_dir = data_dirs[dataset_flag]
    return load_specific_data(data_dir,inds)
