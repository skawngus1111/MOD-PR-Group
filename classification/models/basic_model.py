import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

class Basic_Model(pl.LightningModule) :
    def __init__(self):
        super(Basic_Model, self).__init__()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)

        return loss