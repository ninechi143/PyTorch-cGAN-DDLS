# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from tqdm import tqdm

image_shape = [1,28,28] #size for mnist = [28,28,1]

class Generator(nn.Module):
    def __init__(self, latent_dim = 100, embedding_len = 16):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.embedding_len = embedding_len

        self.label_embedding = nn.Embedding(10, self.embedding_len)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.embedding_len, 128, normalize = False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        label_emb = self.label_embedding(labels)
        latents = torch.cat([noise, label_emb], -1)
        image =  self.model(latents)
        image = image.view(image.size(0), *image_shape)
        return image



class Discriminator(nn.Module):
    def __init__(self, embedding_len = 16):
        super(Discriminator, self).__init__()

        self.embedding_len = embedding_len

        self.label_embedding = nn.Embedding(10, self.embedding_len)
        self.model = nn.Sequential(
            nn.Linear(self.embedding_len + int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        label_emb = self.label_embedding(labels)
        input_concat = torch.cat([img.view(img.size(0), -1), label_emb], -1)
        validity = self.model(input_concat)
        return validity



class cGAN(nn.Module):

    def __init__(self, device = None, generator = None, discrimintor = None, SGLD_lr = 1., SGLD_std = 0.01, SGLD_step = 20):

        super().__init__()
        
        self.device = device

        self.generator = generator
        self.discriminator = discrimintor
        # self.initialize()


        loc = torch.zeros(generator.latent_dim).to(device)
        covariance = torch.eye(generator.latent_dim).to(device)
        
        self.multivariate_normal = torch.distributions.MultivariateNormal(loc = loc, covariance_matrix = covariance, validate_args = False)

        self.SGLD_lr = SGLD_lr
        self.SGLD_std = SGLD_std
        self.SGLD_step = SGLD_step

        
    # def initialize(self):
    #     c = 0
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             init.normal_(m.weight, mean = 0.0 , std=0.01)
    #             init.zeros_(m.bias)
    #             c += 1
    #     print(f"model initialization, #modules: {c}")

    
    def minus_energy_score(self, z = None, label = None):

        logP0_z = self.multivariate_normal.log_prob(z)

        fake_data = self.generator(z, label)
        discriminator_driven_score = self.discriminator(fake_data, label)

        return logP0_z + discriminator_driven_score


    def energy_gradient(self, z, label):
        
        self.generator.eval()
        self.discriminator.eval()

        zi = torch.FloatTensor(z.detach().cpu().data).to(self.device)
        zi.requires_grad = True

        # calculate the gradient
        minus_energy = self.minus_energy_score(zi, label)
        zi_grad = torch.autograd.grad(minus_energy.sum(), [zi], retain_graph=True)[0]

        self.generator.train()
        self.discriminator.train()

        return zi_grad

    def Langevin_Dynamics_step(self, z_old, label, SGLD_lr = None, SGLD_std = None):
        # Calculate gradient wrt x_old
        if SGLD_lr is None: SGLD_lr = self.SGLD_lr
        if SGLD_std is None: SGLD_std = self.SGLD_std

        energy_grad = self.energy_gradient(z_old, label)
        noise = SGLD_std * torch.randn_like(energy_grad).to(self.device)
        z_new = z_old + SGLD_lr * energy_grad + noise
        return z_new


    def generator_DDLS(self, batch_size = 1, latent = None, label = None):
        if latent is None:
            latent = torch.randn(batch_size, self.generator.latent_dim).to(self.device)

        if label is None:
            label = torch.randint(0, 10, (batch_size, )).long().to(self.device)

        for i in range(self.SGLD_step):
            latent = self.Langevin_Dynamics_step(latent, label)

        latent = latent.detach()
        fake_data = self.generator(latent, label)
        return fake_data, label, latent

    
    def generator_sampling(self, batch_size = 1, latent = None, label = None):
        if latent is None:
            latent = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        if label is None:
            label = torch.randint(0, 10, (batch_size, )).long().to(self.device)

        fake_data = self.generator(latent, label)
        return fake_data, label, latent


    def forward(self, x_data = None, label = None):    
        logit = self.discriminator(x_data, label)
        return logit



if __name__ == "__main__":

    # encoder = _Encoder()
    # decoder = _Decoder()

    # # unit-test
    # a_batch_of_images = torch.randn(7 , 1 , 175 , 215)
    # code = encoder(a_batch_of_images)
    # decoded_image = decoder(code)
    # print(code.shape, decoded_image.shape)

    # # Gradient unit-test
    # loss = torch.mean(
    #             torch.sum(
    #                 torch.square(decoded_image - a_batch_of_images) , dim = (1,2,3)
    #             )
    #         )
    # loss.backward()
    # print(loss.item())


    # downstream_task_model unit-test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(100,32).to(device)
    discriminator = Discriminator(32).to(device)

    CGAN = cGAN(device, generator, discriminator, SGLD_lr=1, SGLD_std=0.01, SGLD_step=20,)


    model_parameters = filter(lambda p: p.requires_grad, CGAN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"[!] number of parameters: {params}")

    print(len(CGAN.state_dict()))
    for i in CGAN.state_dict().keys():
        print(i)
