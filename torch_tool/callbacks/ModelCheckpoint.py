import torch

class ModelCheckpoint():
    def __init__(
        self, filepath, monitor='val_loss', verbose=0, 
        save_best_only=False, mode='auto', period=1
    ):
        # ------- parameters -------
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose, self.period = verbose, period

        if save_best_only:
            if mode == 'auto':
                if monitor.find('acc') != -1:
                    self.mode = 'max'
                    self.BestVal = float('-inf')
                else:
                    self.mode = 'min'
                    self.BestVal = float('inf')
            elif mode == 'min':
                self.mode = 'min'
                self.BestVal = float('inf')
            elif mode == 'max':
                self.mode = 'max'
                self.BestVal = float('-inf')
            else:
                raise ValueError('Please reset mode!')

    def check(self, epoch, epochs, log, params):
        # Check the monitor is in the log
        if self.monitor not in log.keys():
            raise ValueError('Please reset the monitor in callback!')
        
        # Callback will save the model in every period
        MonitorVal = log[self.monitor]
        
        if not ((epoch) % self.period):
            file_params = {self.monitor: MonitorVal, 'epoch': epoch}
            filepath = self.filepath.format(**file_params)

            if not self.save_best_only:
                torch.save(params, filepath)
                if self.verbose:
                    print(f"Epoch {epoch:0>5d}: saving model to {filepath}\n")
            else:
                if self.mode == 'min':
                    if MonitorVal < self.BestVal:
                        self.BestVal = MonitorVal
                        self.BestFileParams = {self.monitor: MonitorVal, 'epoch': epoch}
                        self.BestParams = params
                else:
                    if MonitorVal > self.BestVal:
                        self.BestVal = MonitorVal
                        self.BestFileParams = {self.monitor: MonitorVal, 'epoch': epoch}
                        self.BestParams = params

        # save the best model in the last epoch
        if self.save_best_only and (epoch == epochs):
            filepath = self.filepath.format(**self.BestFileParams)
            torch.save(self.BestParams, filepath)
            if self.verbose:
                print(f"Epoch {self.BestFileParams['epoch']:0>5d}: {self.monitor} = {self.BestFileParams[self.monitor]:.4f}, saving model to {filepath}\n")
            self.__reset()
    
    def __reset(self):
        if self.save_best_only:
            if self.mode == 'min':
                self.mode = 'min'
                self.BestVal = float('inf')
            elif self.mode == 'max':
                self.mode = 'max'
                self.BestVal = float('-inf')
            self.BestParams = None