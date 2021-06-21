from torch import nn, optim
from .torch_tool.trainer import TorchModule

class BearingNet(TorchModule):
    def __init__(self, in_channels = 1, encode_size = 40, margin = 1, use_gpu = True, gpu_id = 0):
        super().__init__(use_gpu = use_gpu, gpu_id = gpu_id)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 20, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 20, out_channels = 16, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fully_connected = nn.Sequential(
            nn.BatchNorm1d(144),
            nn.Linear(144, 80),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(80),
            nn.Linear(80, encode_size),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

        self.criterion = nn.TripletMarginLoss(margin = margin)

    def forward(self, img):
        out = self.cnn(img)
        out = self.fully_connected(out)

        return out

    def training_step(self, batch):
        anch, pos, neg, _, _ = batch
        anch_out = self(anch)
        pos_out = self(pos)
        neg_out = self(neg)
        loss = self.criterion(anch_out, pos_out, neg_out)

        return loss

    def val_step(self, batch):
        anch, pos, neg, _, _ = batch
        anch_out = self(anch)
        pos_out = self(pos)
        neg_out = self(neg)
        loss = self.criterion(anch_out, pos_out, neg_out)

        return loss

    def set_optimizer(self):
        optimizer = optim.Adam(
            self.parameters(), lr=0.001, 
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=3e-2, amsgrad=False
        )

        return optimizer