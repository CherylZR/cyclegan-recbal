import os
import argparse
from torch.backends import cudnn
from data_loader import get_loader
from solver import Solver

def main(args):
    # for fast training
    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # data loader
    cloud_loader = get_loader()

    # solver
    solver = Solver(cloud_loader, args)

    # mode selection
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # hyper-parameters


    # training settings
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--gpu', default=0, type=str)
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--resize', default=400, type=str)
    parser.add_argument('--G', default='unet', type=str)
    parser.add_argument('--D', default='patchgan', type=str)
    parser.add_argument('--norm_type', default='instance', type=str)

    # test settings
    parser.add_argument('--test_model', default='20_1000', type=str)

    # paths
    parser.add_argument('--data_path', default='./datasets/cloud', type=str)
    parser.add_argument('--log_path', default='./exp/<time>/log', type=str)
    parser.add_argument('--model_path', default='./exp/<time>/model', type=str)
    parser.add_argument('--sample_path', default='./exp/<time>/sample', type=str)
    parser.add_argument('--result_path', default='./exp/<time>/', type=str)


    args = parser.parse_args()
    print(args)
    main(args)
