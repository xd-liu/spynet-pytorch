#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from dataset import Mdata

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'kitti-final'

# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.size()) not in backwarp_tenGrid:
		tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
		tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, tenInput):
				tenBlue = (tenInput[:, 0:1, :, :] - 0.406) / 0.225
				tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
				tenRed = (tenInput[:, 2:3, :, :] - 0.485) / 0.229

				return torch.cat([ tenRed, tenGreen, tenBlue ], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.netBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)
			# end

			def forward(self, tenInput):
				return self.netBasic(tenInput)
			# end
		# end

		self.netPreprocess = Preprocess()

		self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(__file__.replace('batch_run.py', 'network-' + arguments_strModel + '.pytorch')).items() })
	# end

	def forward(self, tenFirst, tenSecond):
		tenFlow = []

		tenFirst = [ self.netPreprocess(tenFirst) ]
		tenSecond = [ self.netPreprocess(tenSecond) ]

		for intLevel in range(5):
			if tenFirst[0].shape[2] > 32 or tenFirst[0].shape[3] > 32:
				tenFirst.insert(0, torch.nn.functional.avg_pool2d(input=tenFirst[0], kernel_size=2, stride=2, count_include_pad=False))
				tenSecond.insert(0, torch.nn.functional.avg_pool2d(input=tenSecond[0], kernel_size=2, stride=2, count_include_pad=False))
			# end
		# end

		tenFlow = tenFirst[0].new_zeros([ tenFirst[0].shape[0], 2, int(math.floor(tenFirst[0].shape[2] / 2.0)), int(math.floor(tenFirst[0].shape[3] / 2.0)) ])

		for intLevel in range(len(tenFirst)):
			tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tenUpsampled.shape[2] != tenFirst[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tenUpsampled.shape[3] != tenFirst[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tenFlow = self.netBasic[intLevel](torch.cat([ tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
		# end

		return tenFlow
	# end
# end

netNetwork = None

##########################################################

def main():
    global netNetwork
    if netNetwork is None:
        netNetwork = Network().cuda().eval()
        netNetwork = torch.nn.DataParallel(netNetwork).cuda()
    
    intWidth = 720
    intHeight = 1280
    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    list_fn = '/shared/xudongliu/code/semi-flow/hd3/lists/seg_track_val_new_3.txt'
    data_root = '/data5/bdd100k/images/track/val'
    output_dir = 'predictions/bdd-KT-val-3'

    with open(list_fn, 'r') as f:
        fnames = f.readlines()
        assert len(fnames[0].strip().split(' ')) == 2 + args.evaluate
        names = [l.strip().split(' ')[0].split('/')[-1] for l in fnames]
    
    batch_size = 16
    run_dataset = Mdata(data_root, list_fn)
    dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True)
    
    with torch.no_grad():
        for i, (tenPreprocessedFirst, tenPreprocessedSecond) in enumerate(val_loader):
            tenPreprocessedFirst = tenPreprocessedFirst.cuda()
            tenPreprocessedSecond = tenPreprocessedSecond.cuda()
            tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
            tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
            output = tenFlow.cpu()

            curr_bs = output.shape[0]
            for idx in range(curr_bs):
                curr_idx = i * batch_size + idx
                curr_vect = output[idx]
                curr_name = names[curr_idx]
                out_file_name = os.path.join(output_dir, curr_name)
                curr_dir = os.path.split(out_file_name)[0]
                out_file_name = out_file_name.split('.')[0] + '.flo'

                if not os.path.isdir(curr_dir):
                    os.makedirs(curr_dir)

                objOutput = open(out_file_name, 'wb')
                numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
                numpy.array([ curr_vect.shape[2], curr_vect.shape[1] ], numpy.int32).tofile(objOutput)
                numpy.array(curr_vect.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

                objOutput.close()
                
##########################################################

if __name__ == '__main__':
	main()
# end
