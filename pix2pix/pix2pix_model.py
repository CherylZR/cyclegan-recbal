class Pix2PixModel():
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.input_A =
        self.input_B =
        self.netG = networks.defineG
        self.netD = networks.defineD

        if self.isTrain:
            self.
