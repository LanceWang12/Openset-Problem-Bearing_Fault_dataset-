import numpy as np
import sys

#-------------- Logging --------------
def reset_log(log, keys = None):
    if keys is None:
        for key in log.keys():
            log[key] = 0
    else:
        for key in keys:
            log[key] = 0

def mean_log(log, num, keys):
    for key in keys:
        log[key] /= num

def log_to_msg(log, train_end, num, inplace = False, precision=4):
    fmt = "{0}: {1:." + str(precision) + "f}"
    keys = list(log.keys())
    if inplace:
        for k in keys[: train_end]:
            log[k] /= num
        msg = "  ".join(fmt.format(k, log[k]) for k in keys)
    else:
        msg = "  ".join(fmt.format(k, log[k] / num) for k in keys[: train_end])
        msg = msg + '  ' + "  ".join(fmt.format(k, log[k]) for k in keys[train_end:])
    
    return msg

#-------------- Progressbar --------------
class ProgressBar(object):
    def __init__(self, pb_size, length=40):

        # Protect against division by zero
        self.n      = max(1, pb_size)
        self.nf     = float(pb_size)
        self.length = length

        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i/100.0 * pb_size) for i in range(101)])
        self.ticks.add(pb_size-1)

    def print_epochs(self, epochs, epoch):
            print(f'Epoch {epoch + 1}/{epochs} :')
            self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i+1) / self.nf) * self.length))
            p0 = "=" * b
            p1 = " " * (self.length - b)
            p2 = int(100 * ((i + 1) / self.nf))
            sys.stdout.write(f"\r[{p0}{p1}] {p2:3>d}%\t{message}")
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n-1)
        sys.stdout.write(f"{message}\n\n")
        sys.stdout.flush()
        
