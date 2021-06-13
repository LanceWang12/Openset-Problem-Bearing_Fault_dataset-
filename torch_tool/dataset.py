from torch.utils.data import Dataset


class Table(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = len(x)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.size