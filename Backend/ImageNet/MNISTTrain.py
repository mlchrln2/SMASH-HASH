'''Proof of concept for autoencoder based image retrival system'''

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pathlib
from HyperParameters import options
from NeuralNetworks import MNISTAutoencoder


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = options['num_epochs']
batch_size = options['batch_size']

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # normalization is shown to help cnn perform better
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MNISTAutoencoder()

pathlib.Path("./output/output_images").mkdir(parents=True, exist_ok=True)
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        output = model(img)
        loss = model.criterion(output, img)
        # ===================backward====================
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './output/output_images/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './output/MNIST_autoencoder_model.pth')
