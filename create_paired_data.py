# -*- coding: utf-8 -*-
import os
from scipy.misc import imread, imsave
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--folder_A', default='', type=str, help='folder A.')
parser.add_argument('--folder_B', default='', type=str, help='folder B.')
parser.add_argument('--dest', default='', type=str, help='destination folder.')

args = parser.parse_args()

assert args.folder_A != '' and args.folder_B != '' and args.dest != ''

if not os.path.exists(args.dest):
	os.makedirs(args.dest)

A_folder = args.folder_A
B_folder = args.folder_B
A_files = sorted(os.listdir(A_folder))
B_files = sorted(os.listdir(B_folder))

assert len(A_files) == len(B_files) and len(A_files) > 0

for af, bf in zip(A_files, B_files):
	assert af == bf
	print('Processing', af)
	a = imread(os.path.join(A_folder, af), mode='RGB')
	b = imread(os.path.join(B_folder, bf), mode='RGB')
	assert a.shape == b.shape
	combined = np.concatenate([a,b], axis=1)
	imsave(os.path.join(args.dest, af), combined)
