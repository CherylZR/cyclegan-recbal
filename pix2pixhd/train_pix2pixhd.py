# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
import os, sys
sys.path.append('utils')
sys.path.append('models')
from data import *
from network import *
from image_pool import *
from scipy.misc import imsave



class Pix2PixHD():
	def __init__(self, G, D, data, gpu=''):
		self.G = G
		self.D = D
		self.data = data
		self.gpu = gpu
		os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
		self.use_cuda = len(gpu) > 0

		if self.use_cuda:
			self.G.cuda()
			self.D.cuda()

	def numpy2var(self, *arrays):
		vars = []
		for arr in arrays:
			var = Variable(torch.from_numpy(arr))
			if self.use_cuda:
				vars.append(var.cuda())
		return tuple(vars)

	def train(self, sample_dir, ckpt_dir, num_it=100000):
		batch_size = 1
		g_lr = d_lr = 2e-4
		lam_fm = 10
		lam_perceptual = 10

		pool = ImagePool(50)

		optim_G = optim.Adam(self.G.parameters(), lr=g_lr, betas=(0.5, 0.999))
		optim_D = optim.Adam(self.D.parameters(), lr=d_lr, betas=(0.5, 0.999))

		adv_criterion = lambda u,v: torch.mean((u-v)**2) # nn.BCELoss()
		fm_criterion = nn.L1Loss()
		# perceptual_criterion = nn.L1Loss()

		label_true = 1.0
		label_false = 0.0

		if self.use_cuda:
			# adv_criterion = adv_criterion.cuda()
			fm_criterion = fm_criterion.cuda()
			# perceptual_criterion = perceptual_criterion.cuda()

		for it in range(num_it):
			# fetch data
			A, B = self.data(batch_size)
			X_real, Y_real = self.numpy2var(A, B)

			# update D
			feat_real = self.D(torch.cat((X_real, Y_real), dim=1))

			Y_fake = self.G(X)
			feat_fake = self.D(pool.query(torch.cat((X_real, Y_fake.detach()), dim=1)))

			self.optim_D.zero_grad()
			d_loss = 0.0
			for d_real in feat_real:
				d_loss += adv_criterion(d_real[-1], label_true)
			for d_fake in feat_real:
				d_loss += adv_criterion(d_fake[-1], label_false)
			d_loss = d_loss / (len(feat_fake) + len(feat_real))
			d_loss.backward()
			self.optim_D.step()

			# update G
			self.optim_G.zero_grad()
			feat_fake = self.D(torch.cat((X_real, Y_fake), dim=1))
			g_fm_loss = 0.0
			g_adv_loss = 0.0
			g_perceptual_loss = 0.0
			for i in range(len(feat_fake)):
				weight = 1.0 / (len(feat_fake[i]) - 1.0) / len(feat_fake)
				for ff, fr in zip(feat[feat_fake[i][:-1], feat_real[i][:-1]]):
					g_fm_loss += fm_criterion(ff, fr.detach()) * lam_fm * weight
				g_adv_loss += adv_criterion(feat_fake[i][-1], label_true) / len(feat_fake)
			g_loss = g_fm_loss + g_adv_loss + g_perceptual_loss
			g_loss.backward()
			self.optim_G.step()

			if (it % 500 == 0 and it > 0) or (it == num_it-1):
				samples = np.concatenate((A[0], B[0], np.transpose(Y_fake[0].cpu().data.numpy(), [1,2,0])), axis=1)
				imsave(os.path.join(sample_dir, 'it_%s.png') % str(it).zfill(6), samples)
			if (it % 5000 == 0 and it > 0) or (it == num_it-1):
				torch.save(self.G.state_dict(), os.path.join(ckpt_dir, 'it_%s-G.pth') % str(it).zfill(6))
				torch.save(self.D.state_dict(), os.path.join(ckpt_dir, 'it_%s-D.pth') % str(it).zfill(6))


if __name__ == '__main__':
	gpu = '6'
	num_it = 100000
	sample_dir = 'exp/pix2pixHD/samples'
	ckpt_dir = 'exp/pix2pixHD/ckpts'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	data = Cityscapes() # TODO

	input_nc = data.label_size
	output_nc = 3

	G = Coarse2FineGenerator(input_nc, output_nc)
	D = MultiscaleDiscriminator(input_nc=input_nc+output_nc)

	pix2pixhd = Pix2PixHD(G, D, data, gpu)
	pix2pixhd.train(sample_dir, ckpt_dir, num_it)
