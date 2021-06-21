# Convert ndarray to pytorh tensor
from ..dataset import Table
# The class of progressbar and the function to transform log to string
from ..progressbar import ProgressBar, reset_log, log_to_msg

import numpy as np
import torch
from torch import cuda
from torch.nn import Module

class TorchModule(Module):
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

    def fit_dataloader(
        self, TrainLoader, ValLoader = None, 
        epochs = 1, callbacks = None, verbose = False
    ):
        # initialize
        self.callbacks = callbacks
        device = self.device
        optimizer = self.set_optimizer()
        ValFlag = (ValLoader is not None)

        # ------- set history and log -------
        # history: store loss and metrics in every epochs
        # log: store loss and metrics in the epoch
        # train_end: the index of the last metric for training
        history = {'loss': np.zeros(epochs)}
        log = {'loss': 0}

        if ValFlag:
            history['val_loss'] = np.zeros(epochs)
            log['val_loss'] = 0

        self.to(device)
        self.train()
        PB = ProgressBar(len(TrainLoader))
        for epoch in range(epochs):
            if verbose:
                PB.print_epochs(epochs, epoch)

            # reset log to zero
            reset_log(log)

            for i, train_data in enumerate(TrainLoader):
                train_data = (data.to(device) for data in train_data)

                # get  and update the weight of the model
                optimizer.zero_grad()
                loss = self.training_step(train_data).requires_grad_()
                loss.backward()
                optimizer.step()
                

                # Log record the sum of loss. 
                # Log will be divided by the number of the iteration if updating the progress bar.
                log['loss'] += loss.item()

                if ValFlag:
                # Don't record grad to speed up when testing
                    with torch.no_grad():
                        # some function(like dropout) should be disabled
                        self.eval()

                        reset_log(log)
                        for j, val_data in enumerate(ValLoader):
                            val_data = (data.to(device) for data in val_data)
                            loss = self.val_step(val_data)
                            log['val_loss'] += loss
                        
                        # get the mean of the log(about validation data)
                        log['val_loss'] /= j
                        
                        # turn the model to training mode
                        self.train()

                # update the message of the progressbar
                if verbose:
                    msg = log_to_msg(log, 1, i + 1)
                    PB.bar(i, msg)

            # update the message of the progressbar in the end of an epoch
            if verbose:
                msg = log_to_msg(log, 1, i + 1, inplace = True)
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
        self.eval()

        return history

    def training_step(self, batch):
        # User should complete this block
        raise NotImplementedError( "training_step is virtual! User must overwrite it." )


    def val_step(self, batch):
        # User should complete this block
        raise NotImplementedError( "training_step is virtual! Users must overwrite it." )

    def set_optimizer(self):
        # User should complete this block
        raise NotImplementedError( "training_step is virtual! User must overwrite it." )