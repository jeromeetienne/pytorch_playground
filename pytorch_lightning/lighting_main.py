import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as lightning
import torch

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# define the LightningModule
class LitAutoEncoder(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        # define any number of nn.Modules (or use your current ones)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), 
            nn.ReLU(), 
            # nn.Linear(128, 64),
            # nn.ReLU(), 
            nn.Linear(128, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 128), 
            # nn.ReLU(), 
            # nn.Linear(64, 128), 
            nn.ReLU(), 
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=1e-7)
        return optimizer

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

# init the autoencoder
autoencoder_model = LitAutoEncoder()


# setup data
train_dataset = MNIST(os.path.join(__dirname__, "../data"), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
test_dataset = MNIST(os.path.join(__dirname__, "../data"), train=False, download=True, transform=ToTensor())
test_loader = utils.data.DataLoader(test_dataset, batch_size=32)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = lightning.Trainer(max_epochs=20, accelerator='cpu')
trainer.fit(model=autoencoder_model, train_dataloaders=train_loader)

# Get the loss after training
loss = trainer.logged_metrics['train_loss'].item()
print(f"Final training loss: {loss:.4f}")
print(f"=" * 20)


# test the model
trainer.test(model=autoencoder_model, dataloaders=test_loader)

# ###############################################################################

# # load checkpoint
# checkpoint = "./lightning_logs/version_1/checkpoints/epoch=9-step=1000.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder= autoencoder.encoder, decoder=autoencoder.decoder)

# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()

# # embed 4 fake images!
# fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)