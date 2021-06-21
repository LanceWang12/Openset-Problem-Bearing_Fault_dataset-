# Convert ndarray to pytorh tensor
from ..dataset import Table
# The class of progressbar and the function to transform log to string
from ..progressbar import ProgressBar, reset_log, mean_log, log_to_msg
# metrics
from ..metrics import binary_accuracy, categorical_accuracy 

import numpy as np
import torch
from torch import cuda
from torch import optim, nn
from torch.nn import Module
from torch.utils.data import DataLoader


class KerasModule(Module):
    def __init__(self, use_gpu = True, gpu_id = 0):
        super().__init__()
        # Check device
        if use_gpu and (cuda.is_available()):
            self.device = torch.device(f"cuda:{gpu_id}")
            msg = f'| The model use Cuda:{gpu_id} to train. |'
            print(len(msg) * '-')
            print(msg)
            print(len(msg) * '-', '\n')
        else:
            self.device = torch.device('cpu')
            msg ='| The model use cpu to train. |'
            print(len(msg) * '-')
            print(msg)
            print(len(msg) * '-', '\n')
            

    def compile(
        self, optimizer, loss=None, loss_weights=None, metrics = None, 
        sample_weight_mode=None, weighted_metrics=None, target_tensors=None
    ):
        # set optimizer
        self.optimizer = set_optimizer(optimizer, self.parameters)

        # set loss function
        self.__loss = loss

        # set metrics
        self.metrics_dict = set_metrics(metrics)

    def fit_dataloader(
        self, TrainLoader, ValLoader = None, epochs = 1,
        callbacks = None, verbose = True, class_weight = None
    ):
        # set callbacks
        self.callbacks = callbacks

        # set loss function and class weightㄆ
        self.criterion = set_loss(self.__loss, class_weight)

        # start to train
        pb_size= len(TrainLoader)
        history = self.SimpleTraining(
            epochs, TrainLoader, ValLoader,
            verbose, pb_size
        )

        return history

    def fit(
        self, x=None, y=None, batch_size=None, epochs=1, 
        verbose=True, callbacks=None, validation_split=0.0, 
        validation_data=None, shuffle=True, class_weight=None,
        sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
    ):
        # set callbacks
        self.callbacks = callbacks

        # set loss function and class weight
        self.criterion = set_loss(self.__loss, class_weight)

        # set validation flag
        validation_flag = (((validation_split > 0) and (validation_split < 1)) or (validation_data is not None))

        # ------------------ set data -------------------
        # set validation data
        if validation_flag:
            if validation_data is not None:
                x_val, y_val = validation_data
            else:
                train_size = int((1 - validation_split) * len(x))
                x_val, y_val = x[train_size:], y[train_size:]
                x, y = x[:train_size], y[:train_size]
            
            ValDataset = Table(x_val, y_val)
            ValLoader = DataLoader(ValDataset, batch_size = batch_size * 2, shuffle = shuffle, drop_last = True)
        else:
            ValLoader = None
        
        # set train data
        TrainDataset = Table(x, y)
        TrainLoader = DataLoader(TrainDataset, batch_size = batch_size, shuffle = shuffle, drop_last = True)

        # -------------------- start to train --------------------
        pb_size = len(TrainDataset) // batch_size
        history = self.SimpleTraining(
            epochs, TrainLoader, ValLoader,
            verbose, pb_size
        )

        return history

    def evaluate(self):
        pass
        
    def SimpleTraining(
        self, epochs, TrainLoader, ValLoader = None, 
        verbose = True, pb_size = None
    ):
        # prevent using dict to speed up
        device = self.device
        criterion = self.criterion
        optimizer = self.optimizer
        ValFlag = (ValLoader is not None)
        MetricFlag = (self.metrics_dict is not None)

        # ------- set history and log -------
        # history: store loss and metrics in every epochs
        # log: store loss and metrics in the epoch
        # train_end: the index of the last metric for training
        history = {'loss': np.zeros(epochs)}
        log = {'loss': 0}
        train_end = 1
        if MetricFlag:
            # The keys like ['acc', ...], but it don't include loss!
            metrics_key = list(self.metrics_dict.keys())
            train_end += len(metrics_key)
            for key in metrics_key:
                history[key] = np.zeros(epochs)
                log[key] = 0
        else:
            metrics_key = []

        if ValFlag:
            history['val_loss'] = np.zeros(epochs)
            log['val_loss'] = 0
            ValKeys = ['val_loss']
            for key in metrics_key:
                history[f'val_{key}'] = np.zeros(epochs)
                log[f'val_{key}'] = 0
                ValKeys.append(f'val_{key}')

        PB = ProgressBar(pb_size)
        self.to(device)
        for epoch in range(epochs):
            if verbose:
                PB.print_epochs(epochs, epoch)

            # reset log to zero
            reset_log(log)

            for i, (x_batch, y_batch) in enumerate(TrainLoader):
                # put data to gpu memory
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # update the weight of the model
                optimizer.zero_grad()
                pred = self(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                # Log record the sum of loss. 
                # Log will be divided by the number of the iteration if updating the progress bar.
                log['loss'] += loss.item()

                # Don't record grad to speed up when testing
                with torch.no_grad():
                    # some function(like dropout) should be disabled
                    self.eval()

                    if MetricFlag:
                        for metric in metrics_key:
                            log[metric] += self.metrics_dict[metric](y_batch, pred).item()
                    
                    if ValFlag:
                        reset_log(log, ValKeys)
                        for j, (x_val, y_val) in enumerate(ValLoader):
                            pred = self(x_val)
                            loss = criterion(pred, y_val).item()

                            log['val_loss'] += loss
                            
                            for metric in metrics_key:
                                log[f'val_{metric}'] += self.metrics_dict[metric](y_val, pred).item()
                        
                        # get the mean of the log(about validation data)
                        mean_log(log, (j + 1), ValKeys)
                    
                    # turn the model to training mode
                    self.train()

                # update the message of the progressbar
                if verbose:
                    msg = log_to_msg(log, train_end, i + 1)
                    PB.bar(i, msg)

            # update the message of the progressbar in the end of an epoch
            if verbose:
                msg = log_to_msg(log, train_end, i + 1, inplace = True)
                PB.close(msg)
            
            # record log in every epoch
            for key in log.keys():
                history[key][epoch] = log[key]

            # ------- execute function in callbacks -------
            # The try-block prepares for early-stopping
            try:
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback.check(epoch + 1, epochs, log, self.state_dict())
            except SystemError:
                break 
            # if self.callbacks is not None:
            #     for callback in self.callbacks:
            #         callback.check(epoch + 1, epochs, log, self.state_dict())
        
        # move the model from gpu to cpu
        self.to('cpu')

        return history


# Some units
def set_optimizer(optimizer, parameters = None):
    if type(optimizer) == type('str'):
        if optimizer == 'adam':
            optimizer = optim.Adam(parameters())
        elif optimizer == 'RMSprop':
            optimizer = optim.RMSprop(parameters())
        elif optimizer == 'SGD':
            optimizer = optim.SGD(parameters(), lr = 0.001)
        elif optimizer == 'Adagrad':
            optimizer = optim.Adagrad(parameters())
        else:
            raise ValueError("Please reset optimizer!")

    return optimizer

def set_loss(loss, class_weight = None):
    if type(loss) == type('str'):
        loss = loss.lower()
        if loss == 'binarycrossentropy':
            if class_weight is None:
                loss = nn.BCEWithLogitsLoss()
            else:
                loss = nn.BCEWithLogitsLoss(pos_weight = class_weight[0] / class_weight[1])
        elif loss == 'categoricalcrossentropy':
            if class_weight is None:
                loss = nn.CrossEntropyLoss()
            else:
                loss = nn.CrossEntropyLoss(weight = class_weight)
        elif loss == 'mse':
            loss = nn.MSELoss()
        elif loss == 'mae':
            loss = nn.L1Loss()
        else:
            raise ValueError("Please reset loss function!")
    
    return loss

def set_metrics(metrics):
    if type(metrics) != type([]):
            return None
    else:
        metrics_func = dict()
        metrics = [metric.lower() for metric in metrics]
        for metric in metrics:
            if metric == 'binary_acc':
                metrics_func['acc'] = binary_accuracy
            elif metric == 'acc':
                metrics_func['acc'] = categorical_accuracy
            elif metric == 'mse':
                metrics_func['mse'] = nn.MSELoss()
            elif metric == 'mae':
                metrics_func['mae'] = nn.L1Loss()
            else:
                raise ValueError('Please reset metrics!')

        return metrics_func