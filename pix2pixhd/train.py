# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
import os, sys
sys.path.append('utils')
sys.path.append('models')
from data import *
from network import *
from scipy.misc import imsave


class PGGAN():
	def __init__(self, G, D, data, noise, gpu='2'):
		self.G = G
		self.D = D
		self.data = data
		self.noise = noise
		self.gpu = gpu

		os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
		if len(self.gpu) > 0:
			self.G = self.G.cuda()
			self.D = self.D.cuda()

	def numpy2var(self, arr):
		var = Variable(torch.from_numpy(arr))
		if len(self.gpu) > 0:
			var = var.cuda()
		return var

	def train(self):
		bs_map = {2: 64, 3: 64, 4: 32, 5: 32, 6: 32, 7: 16, 8: 8, 9: 4, 10: 2}
		row_map = {2: 16, 3: 16, 4: 8, 5: 8, 6: 8, 7: 8, 8: 4, 9: 2, 10: 2}
		N = 20000
		optim_D = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999))
		optim_G = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
		for R in range(2, 9):  # resolution ranging from 4 to 256
			bs = bs_map[R]
			phases = ['fade_in', 'stabilize'] if R > 2 else ['stabilize']
			resol = 2 ** R
			cumsum = N if R == 2 else 1
			for phase in phases:
				for it in range(N):
					level = R - 1 + min(N, cumsum)/float(N)
					real = self.numpy2var(self.data(bs, resol, level))
					z = self.numpy2var(self.noise(bs))

					self.D.zero_grad()
					fake = self.G(z, level)

					d_real = self.D(real, level)
					d_fake = self.D(fake.detach(), level)

					# update D
					d_loss = 0.5 * (torch.mean((d_real - 1)**2) + torch.mean((d_fake)**2))
					d_loss.backward()
					optim_D.step()

					# update G
					self.G.zero_grad()
					z = self.numpy2var(self.noise(bs))
					fake = self.G(z, level)
					d_fake = self.D(fake, level)
					g_loss = torch.mean((d_fake - 1)**2)
					g_loss.backward()
					optim_G.step()

					# report
					print('[%s|%s], %sx%s, %s, d_loss: %.4f, g_loss: %.4f' % (it, N, resol, resol, phase, d_loss.data[0], g_loss.data[0]))

					if it % 500 == 0 or it == N-1:
						n_row = row_map[R]
						n_col = int(np.ceil(bs / float(n_row)))
						samples = []
						i = j = 0
						for row in range(n_row):
							one_row = []
							# fake
							for col in range(n_col):
								one_row.append(fake[i].cpu().data.numpy())
								i += 1
							# real
							for col in range(n_col):
								one_row.append(real[j].cpu().data.numpy())
								j += 1
							samples += [np.concatenate(one_row, axis=2)]
						samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])
						half = samples.shape[1] // 2
						samples[:,:half,:] = samples[:,:half,:] - np.min(samples[:,:half,:])
						samples[:,:half,:] = samples[:,:half,:] / np.max(samples[:,:half,:])
						samples[:,half:,:] = samples[:,half:,:] - np.min(samples[:,half:,:])
						samples[:,half:,:] = samples[:,half:,:] / np.max(samples[:,half:,:])
						imsave('exp/PGGAN/samples/%sx%s-%s-epoch_%s.png'%(resol, resol, phase, str(it).zfill(5)), samples)
					if it % 5000 == 0 or it == N-1:
						torch.save(self.D.state_dict(), 'exp/PGGAN/ckpts/%sx%s-%s-epoch_%s-D.pth'%(resol, resol, phase, str(it).zfill(5)))
						torch.save(self.G.state_dict(), 'exp/PGGAN/ckpts/%sx%s-%s-epoch_%s-G.pth'%(resol, resol, phase, str(it).zfill(5)))

					cumsum += 1


latent_size = 512
target_resol = 256
if not os.path.exists('exp/PGGAN/samples'):
	os.makedirs('exp/PGGAN/samples')
if not os.path.exists('exp/PGGAN/ckpts'):
	os.makedirs('exp/PGGAN/ckpts')


G = DeconvGenerator(latent_size, target_resol=target_resol, output_act='tanh', norm='pixel')
D = ConvDiscriminator(input_resol=target_resol, norm='batch')
celeba = CelebA()
noise = RandomNoiseGenerator(latent_size, 'gaussian')

pggan = PGGAN(G, D, celeba, noise, '0')
pggan.train()

