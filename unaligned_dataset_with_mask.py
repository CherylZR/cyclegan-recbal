import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class UnalignedDatasetMask(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_A = os.path.join(opt.dataroot, 'coco_bike')
        self.dir_MA = os.path.join(opt.dataroot, 'coco_bike_ann')
        self.dir_B = os.path.join(opt.dataroot, 'voc_bike')
        self.dir_MB = os.path.join(opt.dataroot, 'voc_bike_ann')

        self.A_paths = make_dataset(self.dir_A)
        self.MA_paths = make_dataset(self.dir_MA)
        self.B_paths = make_dataset(self.dir_B)
        self.MB_paths = make_dataset(self.dir_MB)

        self.A_paths = sorted(self.A_paths)
        self.MA_paths = sorted(self.MA_paths)
        self.B_paths = sorted(self.B_paths)
        self.MB_paths = sorted(self.MB_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.osize = [opt.fineSize, opt.fineSize]


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        MA_path = self.MA_paths[index_A]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        MB_path = self.MB_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        MA_img = Image.open(MA_path).convert('L')

        B_img = Image.open(B_path).convert('RGB')
        MB_img = Image.open(MB_path).convert('L')


        transform_img = transforms.Compose([
            transforms.Scale(self.osize, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        transform_mask = transforms.Compose([
            transforms.Scale(self.osize, Image.BICUBIC),
            transforms.ToTensor()
        ])

        A = transform_img(A_img)
        MA = transform_mask(MA_img)
        B = transform_img(B_img)
        MB = transform_mask(MB_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # if input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #     A = tmp.unsqueeze(0)
        #
        # if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'MA': MA, 'MB': MB,
                'A_paths': A_path, 'B_paths': B_path, 'MA_paths': MA_path, 'MB_paths': MB_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDatasetMask'
