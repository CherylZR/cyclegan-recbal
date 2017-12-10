import torch
from model import *
from data_loader import DataLoader


class Solver():

    def __init__(self, data_loader, args):
        # data loader
        self.data_loader = DataLoader(args)
        self.dataset = self.data_loader.load_data()

        # model hyper-parameters
        self.lr_g = args.lr_g
        self.lr_d = args.lr_d
        self.beta1 = args.beta1
        self.beta2 = args.beta2


        # training settings
        self.

        # test settings
        self.test_model = args.test_model

        # paths
        self.log_path = args.log_path
        self.model_path = args.model_path
        self.sample_path = args.sample_path
        self.result_path = args.result_path

        # step size
        self.

        # build model
        self.build_model()

    def build_model(self):
        self.G = Unet()                           #
        self.D = PatchGAN()                       #

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr_g, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2))

        self.G.cuda()
        self.D.cuda()

    def set_input(self, input):
        input_A = input['A']                      #???
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def train(self):
        iters_per_epoch = len(self.data_loader)
