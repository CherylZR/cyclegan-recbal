import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=str)
parser.add_argument('--batch_size', default=1, type=str)
parser.add_argument('--resize', default=400, type=str)

parser.add_argument('--model_dir', default='./exp/<time>/models', type=str)
parser.add_argument('--sample_dir', default='./exp/<time>/samples', type=str)

parser.add_argument('--G', default='unet', type=str)
parser.add_argument('--D', default='patchgan', type=str)
parser.add_argument('--norm_type', default='instance', type=str)

args = parser.parse_args()


G = args.G
D = args.D



model = pix2pix(G, D, dataset, options)
model.train()
