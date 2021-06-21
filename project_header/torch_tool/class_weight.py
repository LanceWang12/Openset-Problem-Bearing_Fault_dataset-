import torch

def compute_class_num(n_classes: int, y: torch.tensor):
    # cdef int class_num[6]
    class_num = torch.zeros(n_classes)
    for i in range(n_classes):
        class_num[i] = torch.sum(y == i)
    return class_num

def compute_class_weight(class_weight, n_classes: int, y):
    # class_weight: a flag
    # n_classes: How many classes in the dataset
    # y: the label in the dataset
    # return the class weight
    y = torch.as_tensor(y, dtype = torch.float32)
    if  class_weight == 'balanced':
        # class_num: the number of every class in this dataset
        class_num = compute_class_num(n_classes, y)
        class_weight = (1 / class_num) * (len(y)) / 2.0 
    elif class_weight is None:
        class_weight = torch.ones(n_classes)
    #else:
    #    raise ValueError('Please set class weight to balanced or None!')
    return class_weight