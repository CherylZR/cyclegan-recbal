import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class CycleGANMaskModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_MA = self.Tensor(nb, 1, size, size)
        self.input_MB = self.Tensor(nb, 1, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_MA_pool = ImagePool(opt.pool_size)
            self.fake_MB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_MA = input['MA']
        input_B = input['B' if AtoB else 'A']
        input_MB = input['MB']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_MA.resize_(input_MA.size()).copy_(input_MA)
        self.input_MB.resize_(input_MB.size()).copy_(input_MB)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_MA = Variable(self.input_MA)
        self.real_MB = Variable(self.input_MB)


    def test(self):
        # real_A = Variable(self.input_A, volatile=True)
        # fake_B = self.netG_A(real_A)
        # self.rec_A = self.netG_B(fake_B).data
        # self.fake_B = fake_B.data

        fake_MB = self.netG_A(Variable(torch.cat([self.input_A, self.input_MA], 1), volatile=True))
        input_MB = fake_MB.data
        self.rec_MA = self.netG_B(Variable(torch.cat([self.input_A, input_MB], 1))).data
        self.fake_MB = fake_MB.data

        # real_B = Variable(self.input_B, volatile=True)
        # fake_A = self.netG_B(real_B)
        # self.rec_B = self.netG_A(fake_A).data
        # self.fake_A = fake_A.data

        fake_MA = self.netG_B(Variable(torch.cat([self.input_B, self.input_MB], 1), volatile=True))
        input_MA = fake_MA.data
        self.rec_MB = self.netG_A(Variable(torch.cat([self.input_B, input_MA], 1))).data
        self.fake_MA = fake_MA.data


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_MB = self.fake_MB_pool.query(self.fake_MB)
        real = Variable(torch.cat([self.input_A, self.input_MA], 1))
        # print(fake_MB.size())
        # print(self.input_A.size())
        fake = Variable(torch.cat([self.input_A, fake_MB.data], 1))
        loss_D_A = self.backward_D_basic(self.netD_A, real, fake)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_MA = self.fake_MA_pool.query(self.fake_MA)
        real = Variable(torch.cat([self.input_B, self.input_MB], 1))
        fake = Variable(torch.cat([self.input_B, fake_MA.data], 1))
        loss_D_B = self.backward_D_basic(self.netD_B, real, fake)
        self.loss_D_B = loss_D_B.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_MA = self.netG_B(Variable(torch.cat([self.input_A, self.input_MA], 1)))
            loss_idt_A = self.criterionIdt(idt_MA, Variable(self.input_MA)) * lambda_A * lambda_idt

            # G_B should be identity if real_A is fed.
            idt_MB = self.netG_A(Variable(torch.cat([self.input_B, self.input_MB], 1)))
            loss_idt_B = self.criterionIdt(idt_MB, Variable(self.input_MB)) * lambda_B * lambda_idt

            self.idt_MA = idt_MA.data
            self.idt_MB = idt_MB.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_MB = self.netG_A(Variable(torch.cat([self.input_A, self.input_MA], 1)))
        # print(fake_MB.size())
        # print(self.input_A.size())
        pred_fake = self.netD_A(Variable(torch.cat([self.input_A, fake_MB.data], 1)))
        loss_G_A = self.criterionGAN(pred_fake, True)


        # GAN loss D_B(G_B(B))
        fake_MA = self.netG_B(Variable(torch.cat([self.input_B, self.input_MB], 1)))
        pred_fake = self.netD_B(Variable(torch.cat([self.input_B, fake_MA.data], 1)))
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_MA = self.netG_B(Variable(torch.cat([self.input_A, fake_MB.data], 1)))
        loss_cycle_A = self.criterionCycle(rec_MA, self.real_MA) * lambda_A

        # Backward cycle loss
        rec_MB = self.netG_A(Variable(torch.cat([self.input_B, fake_MA.data], 1)))
        loss_cycle_B = self.criterionCycle(rec_MB, self.real_MB) * lambda_B

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        self.fake_MB = fake_MB.data
        self.fake_MA = fake_MA.data
        self.rec_MA = rec_MA.data
        self.rec_MB = rec_MB.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_MA = util.tensor2grayim(self.input_MA)
        fake_MB = util.tensor2grayim(self.fake_MB)
        rec_MA = util.tensor2grayim(self.rec_MA)
        real_B = util.tensor2im(self.input_B)
        real_MB = util.tensor2grayim(self.input_MB)
        fake_MA = util.tensor2grayim(self.fake_MA)
        rec_MB = util.tensor2grayim(self.rec_MB)
        ret_visuals = OrderedDict([('real_A', real_A), ('real_MA', real_MA), ('fake_MB', fake_MB), ('rec_MA', rec_MA),
                                   ('real_B', real_B), ('real_MB', real_MB), ('fake_MA', fake_MA), ('rec_MB', rec_MB)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_MA'] = util.tensor2grayim(self.idt_MA)
            ret_visuals['idt_MB'] = util.tensor2grayim(self.idt_MB)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
