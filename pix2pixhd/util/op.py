import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SNConv2d(nn.Conv2d):
	'''
		2d convolution with spectral normalization[1].
		[1]. Spectral normalization: https://openreview.net/pdf?id=B1QRgziT-
	'''
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, it=1, eps=1e-8):
		super(SNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.it = it
		self.eps = eps
		self.register_buffer('u', torch.randn(self.weight.size(0), 1))

	def forward(self, x):
		# apply spectral norm
		w = self.weight.view(self.weight.size(0), -1)
		u = Variable(self.u)
		for i in range(self.it):
			v = w.t().mm(u)
			v = v / (torch.norm(v) + self.eps)
			u = w.mm(v)
			u = u / (torch.norm(u) + self.eps)
		snorm = torch.sum(u.t().mm(w).mm(v).data)
		self.weight.data.div_(snorm)
		self.u = u.data

		return super(SNConv2d, self).forward(x)


class SNConvTranspose2d(nn.Conv2d):
	'''
		2d convolution transpose with spectral normalization[1].
		[1]. Spectral normalization: https://openreview.net/pdf?id=B1QRgziT-
	'''
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, it=1, eps=1e-8):
		super(SNConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
		self.it = it
		self.eps = eps
		self.register_buffer('u', torch.randn(self.weight.size(0), 1))

	def forward(self, x):
		# apply spectral norm
		w = self.weight.view(self.weight.size(0), -1)
		u = Variable(self.u)
		for i in range(self.it):
			v = w.t().mm(u)
			v = v / (torch.norm(v) + self.eps)
			u = w.mm(v)
			u = u / (torch.norm(u) + self.eps)
		snorm = torch.sum(u.t().mm(w).mm(v).data)
		self.weight.data.div_(snorm)
		self.u = u.data

		return super(SNConvTranspose2d, self).forward(x)	


class SNLinear(nn.Linear):
	'''
		Linear with spectral normalization[1].
		[1]. Spectral normalization: https://openreview.net/pdf?id=B1QRgziT-
	'''
	def __init__(self, in_features, out_features, bias=True, it=1):
		super(SNLinear, self).__init__(in_features, out_features, bias)
		self.it = it
		self.register_buffer('u', torch.randn(self.weight.size(0), 1))

	def forward(self, x):
		# apply spectral norm
		w = self.weight
		u = Variable(self.u)
		for i in range(self.it):
			v = w.t().mm(u)
			v = v / (torch.norm(v) + 1e-12)
			u = w.mm(v)
			u = u / (torch.norm(u) + 1e-12)
		snorm = torch.sum(u.t().mm(w).mm(v).data)
		self.weight.data.div_(snorm)
		self.u = u.data

		return super(SNLinear, self).forward(x)


class LNConv2d(nn.Conv2d):
	'''
		Conv2d with Layer normalization[1]
		[1]. Layer normalization: https://arxiv.org/abs/1607.06450
	'''
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-4):
		super(LNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.eps = eps
		self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

	def forward(self, x):
		x = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
		x = x - mean(x, axis=range(1, len(x.size())))
		x = x * 1.0/(torch.sqrt(mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.eps))
		x = x * self.gain
		if self.bias is not None:
			x += self.bias
		return x


class LNConvTranspose2d(nn.ConvTranspose2d):
	'''
		ConvTranspose2d with Layer normalization[1]
		[1]. Layer normalization: https://arxiv.org/abs/1607.06450
	'''
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, eps=1e-4):
		super(LNConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
		self.eps = eps
		self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

	def forward(self, x):
		x = F.conv_transpose2d(x, self.weight, None, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
		x = x - mean(x, axis=range(1, len(x.size())))
		x = x * 1.0/(torch.sqrt(mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.eps))
		x = x * self.gain
		if self.bias is not None:
			x += self.bias
		return x


class LNLinear(nn.Linear):
	'''
		Linear with Layer normalization[1]
		[1]. Layer normalization: https://arxiv.org/abs/1607.06450
	'''
	def __init__(self, in_features, out_features, bias=True, eps=1e-4):
		super(LNLinear, self).__init__(in_features, out_features, bias)
		self.eps = eps
		self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

	def forward(self, x):
		x = F.linear(x, self.weight, None)
		x = x - mean(x, axis=range(1, len(x.size())))
		x = x * 1.0/(torch.sqrt(mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.eps))
		x = x * self.gain
		if self.bias is not None:
			x += self.bias
		return x


class PixelNorm(nn.Module):
	'''
		Pixel normalization[1]
		[1]. Progressive growing of GANs: https://arxiv.org/pdf/1710.10196.pdf
	'''
	def __init__(self, eps=1e-8):
		super(PixelNorm, self).__init__()
		self.eps = eps
	
	def forward(self, x):
		return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-8) ** 0.5

	def __repr__(self):
		return self.__class__.__name__ + '(eps = %s)' % (self.eps) 


class Noop(nn.Module):
	def __init__(self):
		super(Noop, self).__init__()

	def forward(self, x):
		return x


def mean(tensor, axis, **kwargs):
	if isinstance(axis, int):
		axis = [axis]
	for ax in axis:
		tensor = torch.mean(tensor, dim=ax, **kwargs)
	return tensor

def resize(v, so):
	si = list(v.size())
	so = list(so)
	assert len(si) == len(so) and si[0] == so[0]

	# Decrease feature maps.
	if si[1] > so[1]:
		v = v[:, :so[1]]

	# Shrink spatial axes.
	if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
		assert si[2] % so[2] == 0 and si[3] % so[3] == 0
		ks = (si[2] // so[2], si[3] // so[3])
		v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)

	if si[2] < so[2]: 
		assert so[2] % si[2] == 0 and so[2] / si[2] == so[3] / si[3]  # currently only support this case
		v = F.upsample(v, scale_factor=so[2]//si[2], mode='nearest')

	# Increase feature maps.
	if si[1] < so[1]:
		z = torch.zeros((v.shape[0], so[1] - si[1]) + so[2:])
		v = torch.cat([v, z], 1)
	return v
