from utils import *


class Generator(nn.Module):
    def __init__(self, img_size=32, latent_dim=100, channels=1):
        super(Generator, self).__init__()

        if use_gpu:
            torch.cuda.set_device(0)

        self.init_size = img_size // 4
        self.dim = 128
        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.dim*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(self.dim),
        )
        self.conv_blocks1 = nn.Sequential(
                nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.dim, 0.8),
                nn.LeakyReLU(0.2, inplace=True),  # origin
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(self.dim, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),  # origin
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels, affine=False)
        )

    def forward(self, z):  # [batch_size, l00]
        out = self.l1(z)  # [batch_size, 8192]
        out = out.view(out.shape[0], self.dim, self.init_size, self.init_size)  # [batch_size, 128, 8, 8]
        img = self.conv_blocks0(out)  # [batch_size, 128, 8, 8]
        img = nn.functional.interpolate(img,scale_factor=2)  # [batch_size, 128, 16, 16]
        img = self.conv_blocks1(img)  # [batch_size, 128, 16, 16]
        img = nn.functional.interpolate(img,scale_factor=2)    # [batch_size, 128, 32, 32]
        img = self.conv_blocks2(img)  # [batch_size, 1, 32, 32]

        state = self.state_dict()
        return img, state



