# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.grad_normalize as gn



class WGAN_GP_Gen_Loss(nn.Module):

    def __init__(self, net_D = None):

        super(WGAN_GP_Gen_Loss, self).__init__()
        self.Net_D = net_D

    def forward(self, fake_images):

        fake_logits = self.Net_D(fake_images)
        loss =  -1 * torch.mean(fake_logits)

        return loss



class WGAN_GP_Disc_Loss(nn.Module):

    def __init__(self, Net_D = None , coeffient = 10. , device = None):

        super(WGAN_GP_Disc_Loss, self).__init__()
        self.Net_D = Net_D
        self.Lambda = coeffient
        self.device = device

    def forward(self, real_images, fake_images):

        real_logits = self.Net_D(real_images)
        fake_logits = self.Net_D(fake_images)

        alpha = torch.rand(real_images.shape[0] , 1 , 1 ,1).to(self.device)
        interpolate_images = fake_images + alpha * (real_images - fake_images)
        interpolate_logits = self.Net_D(interpolate_images)

        gradients = torch.autograd.grad(outputs = interpolate_logits, inputs = interpolate_images,
                                        grad_outputs = torch.ones(interpolate_logits.size()).to(self.device),
                                        create_graph = True, retain_graph = True, only_inputs=True)[0]

        gradients = gradients.view(gradients.shape[0] , -1)
        gradient_penalty = self.Lambda * ((gradients.norm(2 , 1) - 1) ** 2).mean()

        loss = torch.mean(fake_logits) - torch.mean(real_logits) + gradient_penalty

        return loss


class GNGAN_Disc_Loss(nn.Module):

    def __init__(self, Net_D = None):

        super(GNGAN_Disc_Loss, self).__init__()
        self.Net_D = Net_D
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, real_images, fake_images):

        pred_real = gn.normalize_gradient(self.Net_D , real_images)
        pred_fake = gn.normalize_gradient(self.Net_D , fake_images)

        loss_real = self.loss_fn(pred_real, torch.ones_like(pred_real))
        loss_fake = self.loss_fn(pred_fake, torch.zeros_like(pred_fake))

        return loss_real + loss_fake


class GNGAN_Gen_Loss(nn.Module):

    def __init__(self, Net_D = None):

        super(GNGAN_Gen_Loss, self).__init__()
        self.Net_D = Net_D
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, fake_images):

        pred_fake = gn.normalize_gradient(self.Net_D, fake_images)
        
        loss_fake = self.loss_fn(pred_fake, torch.ones_like(pred_fake))
        
        return loss_fake


class Hinge_Disc_Loss(nn.Module):

    def __init__(self , Net_D = None):

        super(Hinge_Disc_Loss, self).__init__()
        self.Net_D = Net_D

    def forward(self, real_images, fake_images):

        real_logits = self.Net_D(real_images)
        fake_logits = self.Net_D(fake_images)

        G_part = torch.mean(torch.relu(1.0 + fake_logits))
        D_part = torch.mean(torch.relu(1.0 - real_logits))

        return G_part + D_part


class Reconstruction_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, label, model_output):

        loss = torch.mean(
                    torch.sum(
                        torch.square(label - model_output) , dim = (1,2,3)
                    )
                )
        return loss