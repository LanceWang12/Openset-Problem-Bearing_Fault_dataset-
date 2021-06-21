class EarlyStopping():
    def __init__(
        self, monitor="val_loss", min_delta=0,
        patience=1, verbose=0, mode="auto",
        baseline=None, restore_best_weights = False
    ):
        # ------- parameters -------
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

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

        self.counter = 0
        self.BaselineFlag = (baseline is not None)
        self.OverBaseline  = False

    def check(self, epoch, epochs, log, params):
        # Check the monitor is in the log
        if self.monitor not in log.keys():
            raise ValueError('Please reset the monitor in callback!')

        # epochs is trash...
        MonitorVal = log[self.monitor]
        
        if self.mode == 'min':
            # User sets baseline and the MonitorValue has not exceeded baseline
            if self.BaselineFlag and (not self.OverBaseline):
                if self.baseline > MonitorVal:
                    self.OverBaseline = True
                    self.counter = 0
                else:
                    self.counter += 1
                if self.BestVal > MonitorVal:
                    self.BestVal = MonitorVal
                    self.BestParams = params
            else:
                if (self.BestVal - MonitorVal) > self.min_delta:
                    self.counter = 0
                else:
                    self.counter += 1
                if self.BestVal > MonitorVal:
                    self.BestVal = MonitorVal
                    self.BestParams = params
        else:
            # User sets baseline and the MonitorValue has not exceeded baseline
            if self.BaselineFlag and (not self.OverBaseline):
                if self.baseline < MonitorVal:
                    self.OverBaseline = True
                    self.counter = 0
                else:
                    self.counter += 1
                if self.BestVal < MonitorVal:
                    self.BestVal = MonitorVal
                    self.BestParams = params
            else:
                if (MonitorVal - self.BestVal) > self.min_delta:
                    self.counter = 0
                else:
                    self.counter += 1
                if self.BestVal < MonitorVal:
                    self.BestVal = MonitorVal
                    self.BestParams = params

        if self.counter == self.patience:
            if self.restore_best_weights:
                copy_model_params(params, self.BestParams)
            if self.verbose:
                print(f'\nEpoch {epoch:0>5d}: early stopping')
                if self.BaselineFlag:
                    baseline_err = abs(MonitorVal - self.baseline) / self.baseline
                    print(f'Baseline error: {baseline_err:.2%}')
            print()
            self.__reset()
            raise SystemError('Early stopping')

    def __reset(self):
        if self.mode == 'min':
            self.mode = 'min'
            self.BestVal = float('inf')
        else:
            self.mode = 'max'
            self.BestVal = float('-inf')
        self.counter = 0
        self.OverBaseline  = False
        self.BestParams = None
            
    
def copy_model_params(dict1, dict2):
        for key in dict2.keys():
            dict1[key] = dict2[key]