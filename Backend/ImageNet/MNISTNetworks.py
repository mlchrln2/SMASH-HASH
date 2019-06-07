'''This module defines the encoder and decoder neural networks for the MNIST Dataset'''

class MNISTDecoder(nn.Module):
    '''Sample decoder network testing on MNIST'''

    def __init__(self):
        super(MNISTDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class MNISTAutoencoder(nn.Module):
    '''Sample autoencoder network testing on MNIST'''

    def __init__(self):
        super(MNISTAutoencoder, self).__init__()
        self.learning_rate = OPTIONS['learning_rate']
        self.encoder = MNISTEncoder()
        self.decoder = MNISTDecoder()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                          weight_decay=1e-5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
