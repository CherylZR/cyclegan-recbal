import torch.utils.data
import os.path
from base_dataset import BaseDataset
from image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image

class DataLoader():

    def name(self):
        return DataLoader

    def __init__(self, args):
        self.dataset = AlignedDataset(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = args.batch_size,
            shuffle = True
        )

    def load_data(self):
        return self

    # each data loader should provide a __iter__ and a __len__
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

class AlignedDataset(BaseDataset):
    def __init__(self, args):

        self.args = args
        self.data_path = args.data_path

        self.dir_AB = os.path.join(args.data_path, args.mode)  # ???
        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(agrs.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.args.loadSize * 2, self.args.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.args.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.args.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.args.fineSize,
            w_offset:w_offset + self.args.fineSize]
        B = AB[:, h_offset:h_offset + self.args.fineSize,
            w + w_offset:w + w_offset + self.args.fineSize]

        input_c = self.args.input_c   # from A to B
        output_c = self.args.output_c
