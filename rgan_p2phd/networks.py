import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
sys.path.append('utils')
from op import *
import math

DEBUG = False


class DeconvGenerator(nn.Module):
	def __init__(self, input_size, num_channels=3, num_layers=4, target_resol=None, output_act='tanh', norm='batch', fmap_max=1024, fmap_base=8192, fmap_decay=1):
		super(DeconvGenerator, self).__init__()
		self.input_size = input_size
		self.num_channels = num_channels
		self.num_layers = num_layers
		self.target_resol = target_resol
		self.output_act = output_act.lower()
		self._num_layers = self.num_layers
		if self.target_resol is not None:
			self._num_layers = int(math.log2(self.target_resol)) - 1
			assert 2 ** (self._num_layers+1) == self.target_resol and self.target_resol >= 4
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'layer', 'pixel', 'spectral']
		self.fmap_base = fmap_base
		self.fmap_max = fmap_max
		self.fmap_decay = fmap_decay

		act = nn.LeakyReLU(0.2)
		if self.output_act == 'sigmoid':
			oact = nn.Sigmoid()
		elif self.output_act == 'tanh':
			oact = nn.Tanh()
		else:  # linear
			oact = Noop()

		layers = []
		toRGB = []
		
		nfo = self.input_size
		for l in range(self._num_layers):
			nfi = nfo 
			nfo = self.get_nf(l+1)
			if l == 0:
				ks, s, p, op = 4, 1, 0, 0
			else:
				ks, s, p, op = 3, 2, 1, 1
			if self.norm == 'batch':
				block = nn.Sequential(nn.ConvTranspose2d(nfi, nfo, ks, s, p, op), nn.BatchNorm2d(nfo), act)
			elif self.norm == 'instance':
				block = nn.Sequential(nn.ConvTranspose2d(nfi, nfo, ks, s, p, op), nn.InstanceNorm2d(nfo), act)
			elif self.norm == 'layer':
				block = nn.Sequential(LNConvTranspose2d(nfi, nfo, ks, s, p, op), act)
			elif self.norm == 'spectral':
				block = nn.Sequential(SNConvTranspose2d(nfi, nfo, ks, s, p, op, it=1), act)
			elif self.norm == 'pixel':
				block = nn.Sequential(nn.ConvTranspose2d(nfi, nfo, ks, s, p, op), PixelNorm(), act)
			layers += [block]
			toRGB += [nn.Sequential(nn.Conv2d(nfo, self.num_channels, 1, 1, 0), oact)]

		self.net = nn.ModuleList(layers)
		self.out = nn.ModuleList(toRGB)

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.zero_()
			elif isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.zero_()

	def get_nf(self, stage):
		return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

	def forward(self, x, level=None):
		assert x.size(1) == self.input_size
		x = x.view(x.size(0), x.size(1), 1, 1)
		assert level >= 2
		if level is None:
			level = self._num_layers - 1  # first resolution is 4
		else:
			level -= 2

		min_level, max_level = int(level), int(math.ceil(level))
		min_lw, max_lw = int(level+1) - level, level - min_level
		if DEBUG:
			print('G, level:{}, min_level:{}, max_level:{}, min_lw:{}, max_lw:{}'.format(level, min_level, max_level, min_lw, max_lw))
		out = {'min_level': None, 'max_level': None}
		for i in range(max_level+1):
			x = self.net[i](x)
			if DEBUG:
				print('G, level={}, size={}'.format(i, x.size()))
			if i == min_level:
				out['min_level'] = self.out[min_level](x)
			if i == max_level:
				if max_level == min_level:
					out['max_level'] = out['min_level']
				else:
					out['max_level'] = self.out[max_level](x)
		
		if min_level == max_level:
			res = out['min_level']
		else:
			res = resize(out['min_level'], out['max_level'].size()) * min_lw + out['max_level'] * max_lw
		if DEBUG:
			print('Generator output size: {}'. format(res.size()))
		return res


class UpsampleGenerator(nn.Module):
	def __init__(self):
		super(UpsampleGenerator, self).__init__()
		pass


class ConvDiscriminator(nn.Module):
	def __init__(self, num_channels=3, num_layers=4, input_resol=None, output_act='sigmoid', norm='batch', mbstat_avg=None, fmap_max=1024, fmap_base=8192, fmap_decay=1):
		super(ConvDiscriminator, self).__init__()
		self.num_channels = num_channels
		self.num_layers = num_layers
		self.input_resol = input_resol
		self.output_act = output_act.lower()
		self._num_layers = self.num_layers
		if self.input_resol is not None:
			self._num_layers = int(math.log2(self.input_resol)) - 1
			assert 2 ** (self._num_layers+1) == self.input_resol and self.input_resol >= 4
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'layer', 'spectral', 'pixel']
		self.mbstat_avg = mbstat_avg
		self.fmap_max = fmap_max
		self.fmap_base = fmap_base
		self.fmap_decay = fmap_decay

		iact = act = nn.LeakyReLU(0.2)
		if self.output_act == 'sigmoid':
			oact = nn.Sigmoid()
		else:  # linear
			oact = Noop()

		layers = []
		fromRGB = []
		nfo = self.num_channels
		for i in range(self._num_layers, 0, -1):
			nfi = nfo
			nfo = self.get_nf(i)
			if i == 1:
				ks, s, p = 4, 1, 0
				act = oact
			else:
				ks, s, p = 3, 2, 1
			if self.norm == 'batch':
				frgb = nn.Sequential(nn.Conv2d(self.num_channels, nfi, 1, 1, 0), nn.BatchNorm2d(nfi), act)
				block = nn.Sequential(nn.Conv2d(nfi, nfo, ks, s, p), nn.BatchNorm2d(nfo), act)
			elif self.norm == 'instance':
				frgb = nn.Sequential(nn.Conv2d(self.num_channels, nfi, 1, 1, 0), nn.InstanceNorm2d(nfi), act)
				block = nn.Sequential(nn.Conv2d(nfi, nfo, ks, s, p), nn.InstanceNorm2d(nfo), act)
			elif self.norm == 'spectral':
				frgb = nn.Sequential(SNConv2d(self.num_channels, nfi, 1, 1, 0, it=1), act)
				block = nn.Sequential(SNConv2d(nfi, nfo, ks, s, p, it=1), act)
			elif self.norm == 'layer':
				frgb = nn.Sequential(LNConv2d(self.num_channels, nfi, 1, 1, 0), act)
				block = nn.Sequential(LNConv2d(nfi, nfo, ks, s, p), act)
			elif self.norm == 'pixel':
				frgb = nn.Sequential(nn.Conv2d(self.num_channels, nfi, 1, 1, 0), PixelNorm(), act)
				block = nn.Sequential(nn.Conv2d(nfi, nfo, ks, s, p), PixelNorm(), act)
			fromRGB += [frgb]
			layers += [block]
		layers += [nn.Conv2d(nfo, 1, 1, 1, 0)]

		self.net = nn.ModuleList(layers)
		self.input = nn.ModuleList(fromRGB)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.zero_()

	def get_nf(self, stage):
		return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)
		
	def forward(self, x, level=None):
		if level is None:
			level = 0
		else:
			level = self._num_layers - (int(math.ceil(level)) - 1) + (level - int(level))
		max_level, min_level = int(level), int(math.ceil(level))
		min_lw, max_lw = int(level+1) - level, level - int(level)
		if DEBUG:
			print('D, level:{}, min_level:{}, max_level:{}, min_lw:{}, max_lw:{}'.format(level, min_level, max_level, min_lw, max_lw))
		
		_in = {'input': None, 'min_level': None, 'max_level': None}
		for i in range(max_level, self._num_layers+1):
			if i == max_level:
				_in['input'] = x
				x = self.input[max_level](x)
			if i == min_level:
				if max_level == min_level:
					_in['min_level'] = _in['max_level'] = x
				else:
					_in['max_level'] = x
					_in['min_level'] = self.input[min_level](_in['input'])
				if DEBUG:
					print('D, min_level={}, max_level={}, min_level_size={}, max_level_size={}'.format(min_level, max_level, \
						_in['min_level'].size(), _in['max_level'].size()))
				x = resize(_in['min_level'], _in['max_level'].size()) * min_lw + _in['max_level'] * max_lw
			x = self.net[i](x)
			if DEBUG:
				print('D, level={}, size={}'.format(i, x.size()))

		return x.view(x.size(0), 1)


class AEGenerator(nn.Module):
	def __init__(self):
		super(AEGenerator, self).__init__()
		pass

	def forward(self, x, level=None):
		pass

AEDiscriminator = AEGenerator


class DeconvRNNGenerator(nn.Module):
	def __init__(self, input_size, num_channels=3, nlpb=2, output_act='tanh', norm='batch', fmap_max=1024, fmap_base=1024, fmap_decay=1):
		super(DeconvRNNGenerator, self).__init__()
		self.input_size = input_size
		self.num_channels = num_channels
		self.nlpb = nlpb
		self.output_act = output_act.lower()
		self.output_act in ['tanh', 'sigmoid', 'linear']
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'spectral', 'layer', 'pixel']
		self.fmap_max = fmap_max
		self.fmap_base = fmap_base
		self.fmap_decay = fmap_decay

		oact = nn.Tanh() if self.output_act == 'tanh' else nn.Sigmoid() if self.output_act == 'sigmoid' else Noop()
		act = nn.LeakyReLU(0.2)

		layers = []
		ks, s, p, op = 3, 2, 1, 1
		nfi = self.num_channels
		nfo = self.get_nf(0)
		if self.norm == 'batch':
			layers += [nn.ConvTranspose2d(nfi, nfo, ks, s, p, op), nn.BatchNorm2d(nfo), act]
		elif self.norm == 'instance':
			layers += [nn.ConvTranspose2d(nfi, nfo, ks, s, p, op), nn.InstanceNorm2d(nfo), act]
		elif self.norm == 'spectral':
			layers += [SNConvTranspose2d(nfi, nfo, ks, s, p, op, it=1), act]
		elif self.norm == 'layer':
			layers += [LNConvTranspose2d(nfi, nfo, ks, s, p, op), act]
		elif self.norm == 'pixel':
			layers += [nn.ConvTranspose2d(nfi, nfo, ks, s, p, op), PixelNorm(), act]
		ks, s, p = 3, 1, 1
		for i in range(1, self.nlpb+1):
			nfi = nfo
			nfo = self.get_nf(i)
			if i == self.nlpb:
				act = oact
				nfo = self.num_channels
			if self.norm == 'batch':
				layers += [nn.Conv2d(nfi, nfo, ks, s, p), nn.BatchNorm2d(nfo), act]
			elif self.norm == 'instance':
				layers += [nn.Conv2d(nfi, nfo, ks, s, p), nn.InstanceNorm2d(nfo), act]
			elif self.norm == 'spectral':
				layers += [SNConv2d(nfi, nfo, ks, s, p, it=1), act]
			elif self.norm == 'layer':
				layers += [LNConv2d(nfi, nfo, ks, s, p), act]
			elif self.norm == 'pixel':
				layers += [nn.Conv2d(nfi, nfo, ks, s, p), PixelNorm(), act]
		self.block = nn.Sequential(*layers)
		self.input = nn.Sequential(nn.ConvTranspose2d(self.input_size, self.num_channels, 4, 1, 0, 0), act)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.zero_()

	def get_nf(self, layer):
		return min(int(self.fmap_base / (2.0 ** (layer * self.fmap_decay))), self.fmap_max)

	def forward(self, x, level=1, first=False):
		# min_level = int(level)
		# max_level = int(math.ceil(level))
		# out = {'min_level': None, 'max_level': None}
		level = int(math.ceil(level-2))
		if DEBUG:
			print('G', x.size())
		if first:
			x = x.view(x.size(0), x.size(1), 1, 1)
			x = self.input(x)
		for i in range(level):
			x = self.block(x)
			if DEBUG:
				print('G', x.size())
		return x

# TODO: add DeconvLSTMGenerator and more


class UpsampleRNNGenerator(nn.Module):
	def __init__(self):
		super(UpsampleRNNGenerator, self).__init__()
		pass

	def forward(self, x, level=None):
		pass


class ConvRNNDiscriminator(nn.Module):
	# TODO: how to make it reasonable
	def __init__(self, num_channels=3, nlpb=2, output_act='tanh', end_output_act='sigmoid', norm='batch', fmap_max=1024, fmap_base=1024, fmap_decay=1):
		super(ConvRNNDiscriminator, self).__init__()
		self.num_channels = num_channels
		self.nlpb = nlpb
		self.output_act = output_act.lower()
		assert self.output_act in ['sigmoid', 'linear', 'tanh']
		self.end_output_act = end_output_act.lower()
		assert self.end_output_act in ['sigmoid', 'linear']
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'spectral', 'layer', 'pixel']
		self.fmap_max = fmap_max
		self.fmap_base = fmap_base
		self.fmap_decay = fmap_decay

		act = nn.Sigmoid() if self.output_act=='sigmoid' else nn.Tanh() if self.output_act=='tanh' else Noop()
		eact = nn.Sigmoid() if self.end_output_act=='sigmoid' else Noop()

		layers = []
		ks, s, p = 3, 2, 1
		nfi = self.num_channels
		nfo = self.get_nf(0)
		if self.norm == 'batch':
			layers += [nn.Conv2d(nfi, nfo, ks, s, p), nn.BatchNorm2d(nfo), act]
		elif self.norm == 'instance':
			layers += [nn.Conv2d(nfi, nfo, ks, s, p), nn.InstanceNorm2d(nfo), act]
		elif self.norm == 'spectral':
			layers += [SNConv2d(nfi, nfo, ks, s, p, it=1), act]
		elif self.norm == 'layer':
			layers += [LNConv2d(nfi, nfo, ks, s, p), act]
		elif self.norm == 'pixel':
			layers += [nn.Conv2d(nfi, nfo, ks, s, p), PixelNorm(), act]
		ks, s, p = 3, 1, 1
		for i in range(1, self.nlpb+1):
			nfi = nfo
			nfo = self.get_nf(i)
			if i == self.nlpb:
				nfo = self.num_channels
			if self.norm == 'batch':
				layers += [nn.Conv2d(nfi, nfo, ks, s, p), nn.BatchNorm2d(nfo), act]
			elif self.norm == 'instance':
				layers += [nn.Conv2d(nfi, nfo, ks, s, p), nn.InstanceNorm2d(nfo), act]
			elif self.norm == 'spectral':
				layers += [SNConv2d(nfi, nfo, ks, s, p, it=1), act]
			elif self.norm == 'layer':
				layers += [LNConv2d(nfi, nfo, ks, s, p), act]
			elif self.norm == 'pixel':
				layers += [nn.Conv2d(nfi, nfo, ks, s, p), PixelNorm(), act]
		self.block = nn.Sequential(*layers)
		self.end = nn.Sequential(nn.Linear(self.num_channels*4*4, 1), eact)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.zero_()

	def get_nf(self, layer):
		return min(int(self.fmap_base / (2.0 ** (layer * self.fmap_decay))), self.fmap_max)

	def forward(self, x, level=None, last=False):
		if level is None:
			level = math.log2(x.size(2))
			assert 2**level == x.size(2) and x.size(2) == x.size(3)
		level = int(math.ceil(level))
		if DEBUG:
			print('D', x.size(), 'level', level)
		for i in range(level-2):
			assert x.size(2) >= 8 and x.size(3) >= 8
			x = self.block(x)
			if DEBUG:
				print('D', x.size())
		if last:
			x = x.view(x.size(0), -1)
			x = self.end(x)
			if DEBUG:
				print('D', x.size())
		return x


# Generator of pix2pixHD
class Coarse2FineGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, block='residual', ngf=32, n_downsample_innermost=3, n_blocks_innermost=9, 
				 n_pyramid_exclude_innermost=1, n_blocks_per_pyramid=3, norm='batch', padding_type='reflect', activation=nn.ReLU(True)):
		super(Coarse2FineGenerator, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.block = block
		self.ngf = ngf
		self.n_downsample_innermost = n_downsample_innermost
		self.n_blocks_innermost = n_blocks_innermost
		self.n_pyramid_exclude_innermost = n_pyramid_exclude_innermost
		self.n_blocks_per_pyramid = n_blocks_per_pyramid
		self.norm = norm
		self.padding_type = padding_type
		self.activation = activation
		if self.block != 'residual':
			raise NotImplementedError('%s has not been implemented yet.' % self.block)

		ngf_innermost = self.ngf * 2 ** (self.n_pyramid_exclude_innermost)
		self.innermost_model = DownResUpEncoderDecoder(self.input_nc, self.output_nc, ngf_innermost, self.n_downsample_innermost,\
								self.n_blocks_innermost, self.norm, self.padding_type, self.activation, False, True)

		# pyramid
		self.pyramid = nn.ModuleList()
		ngf_pyramid = ngf_innermost
		for i in range(self.n_pyramid_exclude_innermost):
			ngf_pyramid = ngf_pyramid // 2
			no_last_layer = True if i < self.n_pyramid_exclude_innermost - 1 else False
			self.pyramid.append(DownResUpEncoderDecoder(self.input_nc, self.output_nc, ngf_pyramid, 1, self.n_blocks_per_pyramid, \
								self.norm, self.padding_type, self.activation, False, no_last_layer))

		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

	def forward(self, x):
		input_downsampled = [x]
		for i in range(self.n_pyramid_exclude_innermost):
			input_downsampled.append(self.downsample(input_downsampled[-1]))

		# innermost
		output = self.innermost_model(input_downsampled[-1], 'down+resnet+up')
		# pyramid
		for i in range(self.n_pyramid_exclude_innermost):
			input = input_downsampled[self.n_pyramid_exclude_innermost-i-1]  # get input
			down = self.pyramid[i](input, command='down')  # downsample
			# print(down.size(), output.size())
			output = self.pyramid[i](down+output, command='resnet+up')  # resnet + upsample
		return output


class DownResUpEncoderDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm='batch', padding_type='reflect', activation=nn.ReLU(True), use_dropout=False, no_last_layer=False):
		super(DownResUpEncoderDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.n_downsampling = n_downsampling
		self.n_blocks = n_blocks
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'pixel', 'spectral', 'layer']
		self.padding_type = padding_type.lower()
		self.activation = activation
		self.use_dropout = use_dropout
		self.no_last_layer = no_last_layer

		down = [nn.ReflectionPad2d(3)]  # first layer use reflection padding
		# first layer
		if self.norm == 'batch':
			down += [nn.Conv2d(self.input_nc, self.ngf, kernel_size=7, padding=0), nn.BatchNorm2d(self.ngf)]
		elif self.norm == 'instance':
			down += [nn.Conv2d(self.input_nc, self.ngf, kernel_size=7, padding=0), nn.InstanceNorm2d(self.ngf)]
		elif self.norm == 'pixel':
			down += [nn.Conv2d(self.input_nc, self.ngf, kernel_size=7, padding=0), PixelNorm()]
		elif self.norm == 'spectral':
			down += [SNConv2d(self.input_nc, self.ngf, kernel_size=7, padding=0, it=1)]
		elif self.norm == 'layer':
			down += [LNConv2d(self.input_nc, self.ngf, kernel_size=7, padding=0)]
		down += [self.activation]
		# downsample
		down += [DownBlock(self.ngf, self.n_downsampling, 3, 2, 1, self.norm, activation)]  
		# resnet blocks
		resnet = []
		ngf_res = self.ngf * (2 ** self.n_downsampling)
		for i in range(self.n_blocks):
			resnet += [ResidualBlock(ngf_res, self.padding_type, self.norm, self.activation, self.use_dropout)]  
		# upsample
		up = []
		up += [UpBlock(ngf_res, self.n_downsampling, 3, 2, 1, 1, self.norm, self.activation)]  
		# last layer
		if not self.no_last_layer:
			up += [nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, self.output_nc, kernel_size=7, padding=0), nn.Tanh()]
		self.down = nn.Sequential(*down)
		self.resnet = nn.Sequential(*resnet)
		self.up = nn.Sequential(*up)

	def forward(self, x, command='down+resnet+up'):
		assert command in ['down', 'down+resnet', 'down+resnet+up', 'resnet', 'resnet+up', 'up']
		# print('in', x.size())
		if 'down' in command:
			x = self.down(x)
			# print('down', x.size())
		if 'resnet' in command:
			x = self.resnet(x)
			# print('resnet', x.size())
		if 'up' in command:
			x = self.up(x)
			# print('up', x.size())
		return x


class DownBlock(nn.Module):
	def __init__(self, ngf, n_downsampling=3, kernel_size=3, stride=2, padding=1, norm='batch', activation=nn.ReLU(True)):
		super(DownBlock, self).__init__()
		self.ngf = ngf
		self.n_downsampling = n_downsampling
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'pixel', 'spectral', 'layer']
		self.activation = activation

		layers = []
		nfo = self.ngf
		for i in range(self.n_downsampling):
			nfi = nfo
			nfo = nfi * 2
			if self.norm == 'batch':
				layers += [nn.Conv2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), nn.BatchNorm2d(nfo)]
			elif self.norm == 'instance':
				layers += [nn.Conv2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), nn.InstanceNorm2d(nfo)]
			elif self.norm == 'pixel':
				layers += [nn.Conv2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), PixelNorm()]
			elif self.norm == 'spectral':
				layers += [SNConv2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, it=1)]
			elif self.norm == 'layer':
				layers += [LNConv2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)]
			layers += [self.activation]

		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


class UpBlock(nn.Module):
	def __init__(self, ngf, n_upsampling=3, kernel_size=3, stride=2, padding=1, output_padding=1, norm='batch', activation=nn.ReLU(True)):
		super(UpBlock, self).__init__()
		self.ngf = ngf
		self.n_upsampling = n_upsampling
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.output_padding = output_padding
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'pixel', 'spectral', 'layer']
		self.activation = activation

		layers = []
		nfo = self.ngf
		for i in range(self.n_upsampling):
			nfi = nfo
			nfo = nfi // 2
			if self.norm == 'batch':
				layers += [nn.ConvTranspose2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, \
							output_padding=self.output_padding), nn.BatchNorm2d(nfo)]
			elif self.norm == 'instance':
				layers += [nn.ConvTranspose2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, \
							output_padding=self.output_padding), nn.InstanceNorm2d(nfo)]
			elif self.norm == 'pixel':
				layers += [nn.ConvTranspose2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, \
							output_padding=self.output_padding), PixelNorm()]
			elif self.norm == 'layer':
				layers += [LNConvTranspose2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, \
							output_padding=self.output_padding)]
			elif self.norm == 'spectral':
				layers += [SNConvTranspose2d(nfi, nfo, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, \
							output_padding=self.output_padding, it=1)]
			layers += [self.activation]

		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


class ResidualBlock(nn.Module):
	def __init__(self, ngf, padding_type, norm='batch', activation=nn.ReLU(True), use_dropout=False):
		super(ResidualBlock, self).__init__()
		self.ngf = ngf
		self.padding_type = padding_type.lower()
		assert self.padding_type in ['reflect', 'replicate', 'zero']
		self.norm = norm.lower()
		assert self.norm in ['batch', 'instance', 'pixel', 'spectral', 'layer']
		self.activation = activation
		self.use_dropout = use_dropout

		layers = []
		for i in range(2):
			if self.padding_type == 'reflect':
				layers += [nn.ReflectionPad2d(1)]
			elif self.padding_type == 'replicate':
				layers += [nn.ReplicationPad2d(1)]
			elif self.padding_type == 'zero':
				layers += [nn.ZeroPad2d(1)]

			if self.norm == 'batch':
				layers += [nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=0), nn.BatchNorm2d(self.ngf)]
			elif self.norm == 'instance':
				layers += [nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=0), nn.InstanceNorm2d(self.ngf)]
			elif self.norm == 'pixel':
				layers += [nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=0), PixelNorm()]
			elif self.norm == 'spectral':
				layers += [SNConv2d(self.ngf, self.ngf, kernel_size=3, padding=0, it=1)]
			elif self.norm == 'layer':
				layers += [LNConv2d(self.ngf, self.ngf, kernel_size=3, padding=0)]
			if i == 0:
				layers += [self.activation]
				if self.use_dropout:
					layers += [nn.Dropout(0.5)]

		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return x + self.net(x)


class ConvBlock(nn.Module):
	def __init__(self, ngf, padding_type, norm, activation=nn.ReLU(True), use_dropout=False):
		super(ConvBlock, self).__init__()
		pass



class UpsampleBlock(nn.Module):
	def __init__(self, ngf, padding_type, norm, activation=nn.ReLU(True), upsample='nearest', use_dropout=False):
		super(UpsampleBlock, self).__init__()
		pass


# class MultiScaleDiscriminator(nn.Module):
# 	def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', use_sigmoid=False, num_D=3, getIntermFeat=False):
# 		super(MultiScaleDiscriminator, self).__init__()
# 		pass


# # PatchGAN discriminator
# class NLayerDiscriminator(nn.Module):
# 	def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', use_sigmoid=False, getIntermFeat=False):
# 		super(NLayerDiscriminator, self).__init__()
# 		self.input_nc = input_nc
# 		self.ndf = ndf
# 		self.norm = norm.lower()
# 		self.use_sigmoid = use_sigmoid
# 		self.getIntermFeat = getIntermFeat

# 		pass


class MultiscaleDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
				 use_sigmoid=False, num_D=3, getIntermFeat=False):
		super(MultiscaleDiscriminator, self).__init__()
		self.num_D = num_D
		self.n_layers = n_layers
		self.getIntermFeat = getIntermFeat
	 
		for i in range(num_D):
			netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
			if getIntermFeat:  
				for j in range(n_layers+2):
					setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
			else:
				setattr(self, 'layer'+str(i), netD.model)

		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

	def singleD_forward(self, model, input):
		if self.getIntermFeat:
			result = [input]
			for i in range(len(model)):
				result.append(model[i](result[-1]))
			return result[1:]
		else:
			return [model(input)]

	def forward(self, input): 
		num_D = self.num_D
		result = []
		input_downsampled = input
		for i in range(num_D):
			if self.getIntermFeat:
				model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
			else:
				model = getattr(self, 'layer'+str(num_D-1-i))
			result.append(self.singleD_forward(model, input_downsampled))
			if i != (num_D-1):
				input_downsampled = self.downsample(input_downsampled)
		return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
		super(NLayerDiscriminator, self).__init__()
		self.getIntermFeat = getIntermFeat
		self.n_layers = n_layers

		kw = 4
		padw = int(math.ceil((kw-1.0)/2))
		sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

		nf = ndf
		for n in range(1, n_layers):
			nf_prev = nf
			nf = min(nf * 2, 512)
			sequence += [[
				nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
				norm_layer(nf), nn.LeakyReLU(0.2, True)
			]]

		nf_prev = nf
		nf = min(nf * 2, 512)
		sequence += [[
			nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
			norm_layer(nf),
			nn.LeakyReLU(0.2, True)
		]]

		sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

		if use_sigmoid:
			sequence[-1] += [nn.Sigmoid()]

		if getIntermFeat:
			for n in range(len(sequence)):
				setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
		else:
			sequence_stream = []
			for n in range(len(sequence)):
				sequence_stream += sequence[n]
			self.model = nn.Sequential(*sequence_stream)

	def forward(self, input):
		if self.getIntermFeat:
			res = [input]
			for n in range(self.n_layers+2):
				model = getattr(self, 'model'+str(n))
				res.append(model(res[-1]))
			return res[1:]
		else:
			return self.model(input)

