import glob
import shutil
from os import mkdir
from os.path import isdir
from joblib import dump

import numpy as np
from scipy import io
import torch

# ------- transform the data from *.mat(1d) to *.npy(2d) -------
def read_mat(img_size = 32, class_lst = ['Ball', 'Normal', 'Inner_break', 'Outer_break'], path = './data/Raw'):
    # img_size: the size of the output image
    # class_lst: the name of each class
    # path: the path of raw data

    # create img dir
    data_path = "/".join(path.split('/')[:-1])
    img_dir = data_path + '/Image'
    if not isdir(img_dir):
        mkdir(img_dir)

    # visit all classes
    for Class in class_lst:
        # create class dir of image
        class_dir = img_dir + f'/{Class}'
        if not isdir(class_dir):
            mkdir(class_dir)

        # read data
        print(f'Read {Class}...')
        mat_path = path + '/' + Class + '/*.mat'
        file_lst = glob.glob(mat_path)
        
        # ---- turn the data from ndarray to image ----
        file_idx = 0
        for mat_file in file_lst:
            # -- get target key name --
            num = mat_file.split('/')[-1][0:-4]
            if len(num) < 3:
                num = '0' + num
            # only use the data of drive end
            target_key = f'X{num}_DE_time'
            
            mat = io.loadmat(mat_file)
            arr = mat[target_key].reshape(-1)

            # -- image transform --
            # remove remain data point
            idx = int(arr.shape[0] / (img_size * img_size)) * (img_size * img_size)
            img_arr = arr[:idx].reshape((-1, 1, img_size, img_size))
            for img in img_arr:
                np.save(f'{class_dir}/{file_idx}.npy', img)
                file_idx += 1
            
# ------- train test split -------
def train_test_split(
    test_size, path, 
    class_lst = ['Ball', 'Normal', 'Inner_break', 'Outer_break']
):
    # build the directory of train and test
    train_dir = f'{path}/train'
    if not isdir(train_dir):
        mkdir(train_dir)
        for cla in class_lst:
            mkdir(f'{train_dir}/{cla}')

    test_dir = f'{path}/test'
    if not isdir(test_dir):
        mkdir(test_dir)
        for cla in class_lst:
            mkdir(f'{test_dir}/{cla}')

    train_size = 1 - test_size
    for Class in class_lst:
        class_dir = f'{path}/{Class}'
        # get npy lst
        npy_lst = glob.glob(f'{class_dir}/*.npy')
        split_idx = round(len(npy_lst) * train_size)

        # copy npy to train and test directory
        # train
        for fname in npy_lst[: split_idx]:
            source = fname
            destination = f'{train_dir}/{Class}'
            shutil.move(source, destination)

        # test
        for fname in npy_lst[split_idx: ]:
            source = fname
            destination = f'{test_dir}/{Class}'
            shutil.move(source, destination)

# ------- Get dictionary which record four class data -------
def get_dict(
    path = './data/Image/train', dtype = torch.float32,
    class_lst = ['Normal', 'Inner_break', 'Outer_break', 'Ball']
):
    Dict = dict((cla, []) for cla in class_lst)

    for cla in class_lst:
        file_lst = glob.glob(f'{path}/{cla}/*.npy')
        for file in file_lst:
            Dict[cla].append(np.load(file))
        Dict[cla] = np.asarray(Dict[cla])
        Dict[cla] = torch.as_tensor(Dict[cla], dtype = dtype)

    return Dict

# ------- Get pattern vector -------
def get_pattern(model, path = './data/Image/train', class_lst = ['Normal', 'Inner_break', 'Outer_break', 'Ball']):
    # get train dict
    train_dict = get_dict(path)
    vec_dict = dict()
    model.to('cpu')
    model.eval()

    vec_dir = './data/Pattern_Vector'
    if not isdir(vec_dir):
        mkdir(vec_dir)
    
    with torch.no_grad():
        for cla in class_lst[:3]:
            encoding = model(train_dict[cla])
            mean = torch.mean(encoding, axis = 0)
            vec_dict[cla] = mean

    dump(vec_dict, f'{vec_dir}/PatternVector.pth')

    return vec_dict

