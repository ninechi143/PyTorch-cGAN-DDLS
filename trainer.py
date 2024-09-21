# -*- coding: utf-8 -*-

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.inception import InceptionScore


import numpy as np
import cv2      # for save images
import imageio  # for making GIF
from PIL import Image
import matplotlib.pyplot as plt

from utils.dataset import downstream_task_dataset, collate_fn
from utils.model import cGAN, Generator, Discriminator
from utils.loss import Reconstruction_Loss

import os
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
from datetime import datetime
import shutil


class model_trainer():

    def __init__(self,args):

        self.mode = args.mode
        self.gpu = args.gpu
        
        self.load_ckpt = args.load_ckpt
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.optim = args.optimizer
        self.no_log = args.no_log
        self.note = args.note

        self.DDLS_flag = False
        self.sgld_step = args.sgld_step
        self.sgld_lr = args.sgld_lr
        self.sgld_std = args.sgld_std
        

        self.latent_dim = 100
        self.label_embedding_len = 16


        self.time_slot = datetime.today().strftime("%Y%m%d_%H%M")
        self.logdir = os.path.join(os.path.dirname(__file__), self.time_slot + "_logs" + self.note)
        self.ckpt_dir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok = True)
                    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[!] torch version: {torch.__version__}")
        print(f"[!] computation device: {self.device}, index : {self.gpu}")
        print(f"[!] execution mode: {self.mode}")
    

    def __printer(info):
        def wrap1(function):
            def wrap2(self , *args, **argv):
                print(f"[!] {info}...")
                function(self , *args, **argv)
                print(f"[!] {info} Done.")
            return wrap2
        return wrap1


    @__printer("Data Loading")
    def load_data(self):

        transforms_train = torchvision.transforms.Compose( [
                                    # torchvision.transforms.Lambda(lambda x: 2. * (np.array(x) / 255.) - 1.),
                                    # torchvision.transforms.Lambda(lambda x: torch.from_numpy(x).float()),
                                    # torchvision.transforms.Lambda(lambda x: torch.permute(x, (2,0,1))),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((.5,), (.5,)),
                                    torchvision.transforms.Lambda(lambda x: x + 0.03 * torch.randn_like(x))
                                ])
        
        # transforms_test = torchvision.transforms.Compose( [
        #                             torchvision.transforms.ToTensor(),                            
        #                             torchvision.transforms.Normalize((.5,), (.5,)),
        #                         ])

        
        self.train_dataset = downstream_task_dataset(train_stage=True, transform=transforms_train)    
    

        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, 
                                        shuffle = True, num_workers = 1, collate_fn = collate_fn)
        
                                                             
    @__printer("Setup")
    def setup(self):
        
        # define our model, loss function, and optimizer
        self.log_writer = None
        if self.no_log is False:
            self.log_writer = SummaryWriter(self.logdir)
            self.record_args(self.logdir)
        
        self.generator = Generator(self.latent_dim, self.label_embedding_len).to(self.device)
        self.discriminator = Discriminator(self.label_embedding_len).to(self.device)

        self.CGAN = cGAN(self.device, self.generator, self.discriminator,
                         SGLD_lr=self.sgld_lr, SGLD_std=self.sgld_std, SGLD_step=self.sgld_step,)

        self.show_parameter_size(self.CGAN, "CGAN")

        self.CrossEntropy = torch.nn.BCEWithLogitsLoss(reduction="mean").to(self.device)
        # self.CrossEntropy = torch.nn.MSELoss(reduction="mean").to(self.device)

        

        if self.optim.lower() == "adam":
            self.optimizer_G = torch.optim.Adam(self.CGAN.generator.parameters(), lr=self.lr, betas = (0., 0.999))
            self.optimizer_D = torch.optim.Adam(self.CGAN.discriminator.parameters(), lr=self.lr, betas = (0., 0.999))
        else:
            self.optimizer_G = torch.optim.SGD(self.CGAN.generator.parameters(), lr=self.lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
            self.optimizer_D = torch.optim.SGD(self.CGAN.discriminator.parameters(), lr=self.lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
        
        
        def warmup_cosine_annealing(step, total_steps, lr_max, lr_min):
            warm_up_iter = 1000
            if step < warm_up_iter:
                return step / warm_up_iter
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, 
                        lr_lambda=lambda step: warmup_cosine_annealing(step, self.epochs * len(self.train_loader),
                                                                1,  1e-6 / self.lr))# since lr_lambda computes multiplicative factor
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, 
                        lr_lambda=lambda step: warmup_cosine_annealing(step, self.epochs * len(self.train_loader),
                                                                1,  1e-6 / self.lr))# since lr_lambda computes multiplicative factor

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = "min" , factor = 0.5, patience = 20,  min_lr = 1e-6)

        self.InceptionScore_Estimator = InceptionScore()


        # load checkpoint file to resume training
        if self.load_ckpt:
            self.load()



    def execute(self):

        if self.mode == "train":
            self.train()
            self.save("train_end")
        
        if self.mode == "train_DDLS":
            self.DDLS_flag = True
            self.train()
            self.save("train_DDLS_end")



    @__printer("Model Training")
    def train(self,):
        
        avg_time = 0
        counter_for_logging = 0
        # evaluation metrics
        self.best_epoch = 0
        self.best_inception_score = -1e8
        
        self.state = {"loss_G":[], "loss_D":[], "inception_score":[],}

        for epoch in range(self.epochs):

            st = perf_counter()
            
            self.CGAN.train()
            loss_G, loss_D = 0, 0
            for i , batch, in tqdm(enumerate(self.train_loader), desc="Train Progress", leave=False):

                #-------------------------------------------------------
                # Train Generator
                if self.DDLS_flag:
                    fake_data, fake_labels, _ = self.CGAN.generator_DDLS(batch_size = self.batch_size)
                else:
                    fake_data, fake_labels, _ = self.CGAN.generator_sampling(batch_size = self.batch_size)
                fake_logits = self.CGAN(fake_data, fake_labels)                
                targets = torch.ones([self.batch_size, 1]).to(self.device)
        
                loss_generator = self.CrossEntropy(fake_logits, targets)

                self.optimizer_G.zero_grad()
                loss_generator.backward()
                self.optimizer_G.step()
                self.scheduler_G.step()

                loss_G += (loss_generator.item() / len(self.train_loader))
                # loss_G = 0.9 * loss_G + 0.1 * loss_generator.item()

                #-------------------------------------------------------
                # Train Discriminator
                real_data, real_labels = batch[0].to(self.device), batch[1].to(self.device)
                real_logits = self.CGAN(real_data, real_labels)
                real_targets = torch.ones([real_data.size(0), 1]).to(self.device)

                if self.DDLS_flag:
                    fake_data, fake_labels, _ = self.CGAN.generator_DDLS(batch_size = real_data.size(0))
                else:
                    fake_data, fake_labels, _ = self.CGAN.generator_sampling(batch_size = real_data.size(0))
                fake_logits = self.CGAN(fake_data, fake_labels)                
                fake_targets = torch.zeros([fake_data.size(0), 1]).to(self.device)


                loss_discriminator = self.CrossEntropy(real_logits, real_targets) + self.CrossEntropy(fake_logits, fake_targets)

                self.optimizer_D.zero_grad()
                loss_discriminator.backward()
                self.optimizer_D.step()
                self.scheduler_D.step()

                loss_D += (loss_discriminator.item() / len(self.train_loader))
                # loss_D = 0.9 * loss_D + 0.1 * loss_discriminator.item()

                #-------------------------------------------------------
                counter_for_logging += 1
                if counter_for_logging % 400 == 1:
                    self.train_logging(counter_for_logging)
                
            inception_score = self.inception_score_logging()
            # inception_score = 0

            self.state["inception_score"].append(inception_score)
            self.state["loss_G"].append(loss_G)
            self.state["loss_D"].append(loss_D)

            self.log_writer.add_scalar(f"Loss G" , loss_G , epoch)
            self.log_writer.add_scalar(f"Loss D" , loss_D , epoch)
            self.log_writer.add_scalar(f"Inception Score" , inception_score , epoch)
        

            avg_time = avg_time + (perf_counter() - st - avg_time) / (epoch+1)
            print(f"[!] ┌── Epoch: [{epoch+1}/{self.epochs}] done, Training time per epoch: {avg_time:.3f}")
            print(f"[!] ├── Loss G: {loss_G:.6f}, Loss D: {loss_D:.6f}")
            print(f"[!] ├── Inception Score: {inception_score:.6f}")
            print(f"[!] └──────────────────────────────────────────────────────────────\n")

            if inception_score >= self.best_inception_score:
                self.best_inception_score = inception_score
                self.best_epoch = epoch
                # self.save(f"best_{epoch:04d}")
                self.save(f"best")

                 
            if epoch % 10 == 0:
                print(f"[!] Best Epoch: {self.best_epoch}, Inception Score: {self.best_inception_score:.6f}, \n") 

        if not self.no_log:
            self.log_writer.close()

        for k, v in self.state.items():
            plt.plot(np.arange(len(v)), np.array(v), color = "r")
            plt.title(k); plt.xlabel("epoch")
            plt.savefig(os.path.join(self.logdir, f"{k}.png")); plt.close()

        print(f"[!] Best Epoch: {self.best_epoch}, Inception Score: {self.best_inception_score:.6f}, \n") 


    @__printer("Model Saving")    
    def save(self , name = ""):

        keys = ["CGAN",]
        values = [self.CGAN.state_dict(),]

        checkpoint = {
            k:v for k, v in zip(keys, values) 
        }
        
        torch.save(checkpoint , os.path.join(self.ckpt_dir, f"ckpt_{name}.pth"))


    @__printer("Model Loading")  
    def load(self):
        checkpoint = torch.load(self.load_ckpt)
        self.CGAN.load_state_dict(checkpoint['CGAN'])


    def train_logging(self, step):

        if not self.no_log:

            path = os.path.join(self.logdir, "_train_logging")
            os.makedirs(path, exist_ok=True)

            self.CGAN.eval()            
            num = 100
            tmp = np.repeat(np.arange(10), 10).reshape(10,10).T.reshape([-1])
            labels = torch.from_numpy(tmp).long().to(self.device)
            if self.DDLS_flag:
                sampled_data, _, _ = self.CGAN.generator_DDLS(batch_size = num, label = labels)            
            else:
                sampled_data, _, _ = self.CGAN.generator_sampling(batch_size = num, label = labels)

            torchvision.utils.save_image(sampled_data,
                                         os.path.join(path, f"sampled_image_{step:08d}.png"),
                                         normalize=True, 
                                         nrow = int(np.sqrt(num)))

            self.log_writer.add_image("sampled data", 
                                      torchvision.utils.make_grid(sampled_data, nrow = int(np.sqrt(num)) , normalize = True), 
                                      step)

            self.CGAN.train()

    
    def inception_score_logging(self):

        if not self.no_log:

            self.CGAN.eval()            
            sample_list = []
            for i in tqdm(range(50), leave=False, desc="sampling for InceptionScore"):
                num = 100                
                if self.DDLS_flag:
                    fake_data, _, _ = self.CGAN.generator_DDLS(batch_size = num)
                else:
                    fake_data, _, _ = self.CGAN.generator_sampling(batch_size = num)
                fake_data = fake_data.detach().cpu()
                fake_data = torch.clamp(255*(fake_data*0.5 + 0.5), 0, 255).byte() #normalize to [0,255]
                fake_data = torch.cat([fake_data, fake_data, fake_data], dim = 1) #make it to RGB image
                sample_list.append(fake_data)
            sample_list = torch.cat(sample_list, dim = 0)

            self.InceptionScore_Estimator.update(sample_list)
            score_mean, sroce_std = self.InceptionScore_Estimator.compute()

            self.CGAN.train()

        return score_mean

    def make_gif(self, path, fps = 10):

        non_sorted_list = list(Path(path).rglob("*.png"))
        if len(non_sorted_list) == 0:
            print(f"[!] No such images in folder to make GIF. Please check")
            return

        png_list = sorted(non_sorted_list)

        process = [cv2.imread(str(i))[:, :, ::-1] for i in png_list]
        imageio.mimsave(os.path.join(self.logdir , f"SGLD_process_demo.gif") , process , fps = fps)
        # [os.remove(i) for i in png_list]
        

    def record_args(self , path):

        source_code_path = os.path.dirname(__file__)
        backup_path = os.path.join(path, "src_backup")
        os.makedirs(backup_path, exist_ok=True)
        shutil.copy(os.path.join(source_code_path, "trainer.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "main.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "dataset.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "loss.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "model.py"), backup_path)

        with open(os.path.join(path , "command_args.txt") , "w") as file:
            file.write(f"mode = {self.mode}\n")
            file.write(f"gpu = {self.gpu}\n")
            file.write(f"load ckpt = {self.load_ckpt}\n")
            file.write(f"learning rate = {self.lr}\n")
            file.write(f"optimizer = {self.optim}\n")
            file.write(f"batch size = {self.batch_size}\n")
            file.write(f"epochs = {self.epochs}\n")
            file.write(f"sgld step = {self.sgld_step}\n")
            file.write(f"sgld lr = {self.sgld_lr}\n")
            file.write(f"sgld std = {self.sgld_std}\n")
            
        return


    def show_parameter_size(self, model, model_name = "Model"):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"[!] {model_name} - number of parameters: {params}")
