# Pytorch trainer

## 1. How to use trainer to define a deep learning model?

```python
from torch_tool import trainer

class Net(trainer):
    def __init__(self, input_shape, output_shape, use_gpu = False, gpu_id = 0):
        super().__init__(use_gpu = use_gpu, gpu_id = gpu_id)
        self.fc_layers = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_shape),
        )
    
    def forward(self, x):
        output = self.fc_layers(x)
        return output
```

### The deep learning model should **inherit trainer class**, and use *super( ).\_\_init\_\_( )*  to set up gpu device



## 2. How to use callbacks?

```python
from torch_tool.callbacks import ModelCheckpoint, EarlyStopping

directory = './checkpoint'
filename = 'model_{epoch:02d}-{val_loss:.4f}.pth'
if not os.path.isdir(directory):
    os.mkdir(directory)

ckpt1 = ModelCheckpoint(
    filepath = directory + '/' + filename, monitor='val_loss', verbose=1, 
    save_best_only = False, mode='min', period=3
)
ckpt2 = EarlyStopping(
    monitor="val_loss", min_delta=5000,
    patience=4, verbose=1, mode="auto",
    restore_best_weights=True
)
callbacks = [ckpt1, ckpt2]
```

### Take ModelCheckpoint for example. Easyly and clearly, user just set the parameters like Keras.



## 3. How to train the model? (Take binary classification as example)

```python
from torch_tool import compute_class_weight

# prepare data
n_classes = 2
X, Y = make_classification(
    n_samples = 3000, n_features = 30, n_classes = n_classes
)

Y = Y.reshape((-1, 1))
# !!!! turn the data from ndarray to torch.tensor !!!!
X, Y = torch.as_tensor(X, dtype = torch.float32), torch.as_tensor(Y, dtype = torch.float32)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

# prepare model
model = Net(x_train.shape[1], 1)
model.compile(optimizer = 'adam', loss = 'BinaryCrossentropy', metrics = ['binary_acc'])

# set class weight
n_classes = len(np.unique(Y))
class_weight = compute_class_weight('balanced', n_classes, Y)

history = model.fit(
    x_train, y_train, batch_size = 32, shuffle = True, 
    verbose = True, epochs = 20, validation_split = 0.2, 
    class_weight = class_weight, callbacks = callbacks
)
```

### First, user should <font color=#FF0000>turn the data from ndarray to torch.tensor</font>. Second, use the member function of trainer - *compile* (However pytorch is a dynamic module; I just mimic the style of Keras) to set optimizer, loss function and metrics. Finally, just fit the model.

### After training, model.fit will return the history. It is a dictionary like below:

```python
history = {
  'loss': [0.35, 0.28, 0.24, ...],
  'acc': [0.67, 0.73, 0.82, ...]
}
```





