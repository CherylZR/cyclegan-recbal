import os
import argparse
import numpy as np
from scipy import misc

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default = '', type = str )
parser.add_argument('--path_A', default = '', type = str )
parser.add_argument('--path_B', default = '', type = str )

args = parser.parse_args()

if not os.path.exists(args.path_A):
    os.makedirs(args.path_A)

if not os.path.exists(args.path_B):
    os.makedirs(args.path_B)

files = os.listdir(args.data_path)
for i, name in enumerate(files):
    if name[0:3] =='cat':
        img = misc.imread(name, mode = 'RGB')
        misc.imshow(img)
        print name
