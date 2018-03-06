import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
from PIL import Image
from scipy.linalg import block_diag
import pdb
from scipy.misc import imread, imresize
rand = np.random.randint
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

def preprocess_data(data_dir, save_dir,d,num_classes=200):
    # needs to be called only once.
    # data_dir: where data is currently located
    # save_dir: where preprocessed data should be saved
    # d: size to reshape images to. [d,d,3]
    # num_classes = total number of classes in dataset

    import Image
    xs = np.zeros((0, d, d, 3))
    ys_membership = np.zeros((0, num_classes))
    # fill xs:
    print 'filling xs'
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
                img_arr = np.array(im)
                img_arr = imresize(img_arr, (d, d))  # crop/resize.
                if img_arr.shape!=(d,d,3):
                    print 'bw image!'
                    continue
                xs = np.vstack((xs, img_arr[np.newaxis, :, :, :]))
        #membership_vec = np.zeros((1, num_classes))
        #membership_vec[0, curr_class] = 1
        #ys_membership = np.vstack((ys_membership, membership_vec))
        curr_class += 1
        print 'processing xs'
        xs_mean = np.mean(xs,0)
        xs_var = np.mean((xs-xs_mean)**2, 0)
        xs_normalized = (xs-xs_mean)/np.sqrt(xs_var)
        # save xs,ys
        save_dir_ = save_dir + '/class'+str(curr_class)+'.npy'
        print 'saving at',save_dir_
        np.save(save_dir_,xs_normalized)
    return 0


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

loaded_train_data = None # global. data on RAM
def get_bird_train_data2(data_dir,k,n):
    global loaded_train_data
    train_classes = range(1,101)
    perm = np.random.permutation(train_classes)
    classes = perm[range(k)]
    #print 'classes:',classes
    if loaded_train_data is None:
        print 'loading train data'
        loaded_train_data = [np.load(data_dir+"/class"+str(i)+".npy") for i in train_classes]
    loaded_data = [loaded_train_data[c-1] for c in classes]
    loaded_data = [np.random.permutation(class_data)[:n] for class_data in loaded_data] # take random subsample 
    class_szs = [class_data.shape[0] for class_data in loaded_data]
    agreement_islands = [np.ones((sz,sz)) for sz in class_szs]
    ys = block_diag(*agreement_islands) # partition matrix
    #membership_islands = [np.ones((sz,1)) for sz in class_szs]
    #ys_membership = block_diag(*membership_islands) # membership matrix
    xs = np.concatenate(loaded_data,0)
    # center crop:
    # xs = xs[:,35:265,35:235,:] # crop
    # xs = np.array([imresize(mat,(299,299)) for mat in xs]) # resize
    return xs,ys

def augment(data_dir):
    """ this should only be called once """
    for i in range(1,201):
        class_data_path = data_dir+"/class{}.npy".format(str(i))
        class_data = np.load(class_data_path)
        # ad-hoc fix:
        n_class_data = class_data.shape[0]
        mid = n_class_data/2
        class_data = class_data[:mid]
        print i,class_data_path
        class_data_flipped = np.flip(class_data,2) # horizontal flipping
        class_data = np.vstack((class_data,class_data_flipped))
        np.save(class_data_path,class_data)
#augment('/specific/netapp5_2/gamir/carmonda/research/vision/caltech_birds')

def load_specific_data(data_dir,inds,augment=False):
    global loaded_test_data
    def echo(x):
        print 'loading:',x
        return True
    
    data_paths = [data_dir+"/class"+str(i)+".npy" for i in inds]
    loaded_data = [np.load(path) if echo(path) else None for path in data_paths]
    if not augment:
        loaded_data = [class_data[:class_data.shape[0]/2] for class_data in loaded_data] # one half is augmented data
    class_szs = [class_data.shape[0] for class_data in loaded_data]
    agreement_islands = [np.ones((sz,sz)) for sz in class_szs]
    ys = block_diag(*agreement_islands) # partition matrix
    membership_islands = [np.ones((sz,1)) for sz in class_szs]
    ys_membership = block_diag(*membership_islands) # membership matrix
    xs = np.concatenate(loaded_data,0)
    # center crop:
    # xs = xs[:,35:265,35:235,:] # crop
    # xs = np.array([imresize(mat,(299,299)) for mat in xs]) # resize
    return xs,ys,ys_membership

def l2_normalize(arr):
    arr_norms = np.sqrt(np.sum(arr**2,1))
    arr_norms = np.reshape(arr_norms,[arr.shape[0],1])
    return arr/arr_norms
