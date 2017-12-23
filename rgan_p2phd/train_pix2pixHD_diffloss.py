# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
import os, sys, time
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
				var = var.cuda()
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

		adv_criterionD = lambda u,v: torch.mean((u-v)**2) # nn.BCELoss()
		adv_criterionG = lambda u,v: -torch.mean(v*torch.log(u+1e-8) + (1-v)*torch.log(1-u+1e-8))
		fm_criterion = nn.L1Loss()
		# perceptual_criterion = nn.L1Loss()

		label_true = 1.0
		label_false = 0.0

		if self.use_cuda:
			# adv_criterion = adv_criterion.cuda()
			fm_criterion = fm_criterion.cuda()
			# perceptual_criterion = perceptual_criterion.cuda()

		for it in range(num_it):
			begin_time = time.time()
			# fetch data
			A, B = self.data(batch_size)
			X_real, Y_real = self.numpy2var(A, B)
			prepare_time = time.time() - begin_time

			# update D
			feat_real = self.D(torch.cat([X_real, Y_real], dim=1))

			Y_fake = self.G(X_real)
			feat_fake = self.D(pool.query(torch.cat([X_real, Y_fake.detach()], dim=1)))

			optim_D.zero_grad()
			d_loss = 0.0
			for d_real in feat_real:
				d_loss += adv_criterionD(d_real[-1], label_true)
			for d_fake in feat_fake:
				d_loss += adv_criterionD(d_fake[-1], label_false)
			d_loss = d_loss / (len(feat_fake) + len(feat_real))
			d_loss.backward()
			optim_D.step()

			# update G
			optim_G.zero_grad()
			feat_fake = self.D(torch.cat([X_real, Y_fake], dim=1))
			g_fm_loss = 0.0
			g_adv_loss = 0.0
			g_perceptual_loss = 0.0
			for i in range(len(feat_fake)):
				if len(feat_fake[i]) == 1:
					weight = 1
				else:
					weight = 1.0 / (len(feat_fake[i]) - 1.0) / len(feat_fake)
				for ff, fr in zip(feat_fake[i][:-1], feat_real[i][:-1]):
					g_fm_loss += fm_criterion(ff, fr.detach()) * lam_fm * weight
				g_adv_loss += adv_criterionG(feat_fake[i][-1], label_true) / len(feat_fake)
			g_loss = g_fm_loss + g_adv_loss + g_perceptual_loss
			g_loss.backward()
			optim_G.step()

			elapsed_time = time.time() - begin_time - prepare_time

			print('[%s|%s], d_loss: %.4f, g_loss: %.4f, prepare_time: %.4fsec, elapsed_time: %.4fsec' % \
					(it, num_it, d_loss.data[0], g_loss.data[0], prepare_time, elapsed_time))

			if (it % 500 == 0) or (it == num_it-1):
				A_ = np.tile(np.argmax(A[0], axis=0).reshape(1, A[0].shape[1], A[0].shape[2]), [3, 1, 1])
				A_ = ((A_ - A_.min()) / (A_.max() - A_.min()) - 0.5) * 2
				samples = np.transpose(np.concatenate((A_, B[0], Y_fake[0].cpu().data.numpy()), axis=1), [1,2,0])
				imsave(os.path.join(sample_dir, 'it_%s.png') % str(it).zfill(6), samples)
			if (it % 5000 == 0 and it > 0) or (it == num_it-1):
				torch.save(self.G.state_dict(), os.path.join(ckpt_dir, 'it_%s-G.pth') % str(it).zfill(6))
				torch.save(self.D.state_dict(), os.path.join(ckpt_dir, 'it_%s-D.pth') % str(it).zfill(6))


if __name__ == '__main__':
	gpu = '8'
	num_it = 100000
	sample_dir = 'exp/pix2pixHD_diffloss/samples'
	ckpt_dir = 'exp/pix2pixHD_diffloss/ckpts'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	data = Cityscapes() 

	input_nc = data.label_size
	output_nc = 3

	G = Coarse2FineGenerator(input_nc, output_nc)
	D = MultiscaleDiscriminator(input_nc=input_nc+output_nc, getIntermFeat=True, use_sigmoid=True)

	pix2pixhd = Pix2PixHD(G, D, data, gpu)
	pix2pixhd.train(sample_dir, ckpt_dir, num_it)