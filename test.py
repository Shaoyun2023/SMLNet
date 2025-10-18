

import os
import torch
from torch.autograd import Variable
from net import SMLNet,SPDconv
import utils
from args_fusion import args
import numpy as np
# from scipy.misc import imread, imsave, imresize
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

import torchvision
from ptflops import get_model_complexity_info

from thop import profile

def load_model1(path, deepsupervision):
	input_nc = 1
	output_nc = 1
	nb_filter=[16, 64, 32, 16]

	nest_model = SMLNet(nb_filter, input_nc, output_nc, deepsupervision)


	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.to(args.device)


	return nest_model

def load_model2(path, deepsupervision):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	SPD_model = SPDconv(in_chans=input_nc, out_chans=output_nc)




	SPD_model.load_state_dict(torch.load(path), False)

	para = sum([np.prod(list(p.size())) for p in SPD_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(SPD_model._get_name(), para * type_size / 1000 / 1000))

	SPD_model.eval()
	# SPD_model.cuda()

	return SPD_model


def run_demo(nest_model, SPD_model, infrared_path, visible_path, output_path_root, index, f_type):
	img_ir, h, w, c = utils.get_test_image1(infrared_path)
	img_vi, h, w, c = utils.get_test_image1(visible_path)



	if c is 0:
		if args.cuda:
			img_ir = img_ir.to(args.device)
			img_vi = img_vi.to(args.device)
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)

		x11 = SPD_model.spdconv(img_ir.cpu(), img_vi.cpu())[0]
		x12 = SPD_model.spdconv(img_ir.cpu(), img_vi.cpu())[1]

		x11 = SPD_model.spdconv(x11, x12)[0]
		x12 = SPD_model.spdconv(x11, x12)[1]

		x11 = SPD_model.spdconv(x11, x12)[0]
		x12 = SPD_model.spdconv(x11, x12)[1]
		#
		# x11 = x11.to(args.device)
		# x12 = x12.to(args.device)

		bbb = nest_model.encodertrain(img_vi, img_ir)[0]

		img_fusion = bbb


	else:
		img_fusion_blocks = []
		img_fusion_list = []
		for i in range(c):
			img_vi_temp = img_vi[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vi_temp = img_vi_temp.to(args.device)
				img_ir_temp = img_ir_temp.to(args.device)
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)


			spdd = SPD_model.spdconv(img_ir_temp.cpu(), img_vi_temp.cpu())


			x11 = spdd[0]
			x12 = spdd[1]


			x11 = x11.to(args.device)
			x12 = x12.to(args.device)

			bbb = nest_model.encodertrain(x11, x12)[0]


			# x_min = torch.min(bbb)
			# x_max = torch.max(bbb)
			# bbb = (bbb - x_min) / (x_max - x_min) * 255


			# bbb = -bbb

			# x_min2 = torch.min(bbb)
			# x_max2 = torch.max(bbb)
			# x = (bbb- x_min2) / (x_max2 - x_min2) * 255
			# img_fusion = ((bbb / 2) + 0.5) * 255
			img_fusion_blocks.append(bbb)
		print(h, "h")
		print(w, "w")
		if h == 256 and w == 256:
			img_fusion_list = utils.recons_fusion_images11(img_fusion_blocks, h, w)
		if 256 < h <= 512 and 256 < w <= 512:
			img_fusion_list = utils.recons_fusion_images1(img_fusion_blocks, h, w)

		if h== 576 and w == 768:
			img_fusion_list = utils.recons_fusion_images576768(img_fusion_blocks, h, w)

		if h == 450  and w == 620:
			img_fusion_list = utils.recons_fusion_images450620(img_fusion_blocks, h, w)

		if 512 < h <= 768 and 512 < w <= 768:
			img_fusion_list = utils.recons_fusion_images2(img_fusion_blocks, h, w)
		if 512 < h < 768 and 768 < w <= 1024:
			img_fusion_list = utils.recons_fusion_images3(img_fusion_blocks, h, w)

		if 256 < h < 512 and 512 < w < 768 and h!= 450 and w != 620:
			img_fusion_list = utils.recons_fusion_images4(img_fusion_blocks, h, w)
		if 768 <= h <= 1024 and 1024 <= w <= 1280:
			img_fusion_list = utils.recons_fusion_images5(img_fusion_blocks, h, w)
		if 0 < h < 256 and 256 < w < 512:
			img_fusion_list = utils.recons_fusion_images6(img_fusion_blocks, h, w)
		if 0 < h < 256 and 512 < w < 768:
			img_fusion_list = utils.recons_fusion_images7(img_fusion_blocks, h, w)
		if h == 256 and 512 < w < 768:
			img_fusion_list = utils.recons_fusion_images8(img_fusion_blocks, h, w)
	# img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)
	# image = (img_fusion_list[0]).squeeze().detach().cpu().numpy()
	# image = (image - image.min()) / (image.max() - image.min())

	# plt.imshow((img_fusion_list[0]).cpu().squeeze(), cmap='jet')
	# plt.axis('off')
	#
	# plt.savefig('figure/1.png', bbox_inches='tight', pad_inches=0.0)
	#
	# plt.show()

	output_count = 0
	for img_fusion in img_fusion_list:

		# image = img_fusion.squeeze().detach().cpu().numpy()
		#
		# image = (image - image.min()) / (image.max() - image.min())
		# plt.imshow(image, cmap='jet')
		# # plt.imshow(image, cmap='gray')
		# plt.axis('off')
		# plt.show()

		# print(img_fusion)
		# x_min2 = torch.min(img_fusion)
		# x_max2 = torch.max(img_fusion)
		# img_fusion = (img_fusion- x_min2) / (x_max2 - x_min2) * 255
		# img_fusion = -img_fusion
		if index < 10:
			file_name = '0' + str(index) + '.png'
		else:
			file_name = str(index) + '.png'
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


def main():

	# run demo
	test_path = "images/test-RoadScene/"
	network_type = 'SwinFuse'
	fusion_type = ['l1_mean']

	output_path = 'outputs/attention_avg/'

	# in_c = 3 for RGB imgs; in_c = 1 for gray imgs
	in_chans = 1

	num_classes = in_chans
	mode = 'L'
	model_path = args.model_default

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[1])
		ssim_weight_str = args.ssim_path[1]
		f_type = fusion_type[0]

		model1 = load_model1(model_path, num_classes)
		model2 = load_model2(model_path, num_classes)

		# input = torch.randn(1, 1, 256, 256)
		# flops, params = profile(model2, inputs=(input,))
		#
		# print(f"FLOPs: {flops / 1e9:.4f} G")
		# print(f"Params: {params / 1e6:.4f} M")

		# begin = time.time()
		# for a in range(10):
		for i in range(362):
			# for i in range(1000, 1221):
			# for i in range(1000, 1040):
			index = i + 1
			infrared_path = test_path + 'IR' + str(index) + '.png'
			visible_path = test_path + 'VIS' + str(index) + '.png'
			# infrared_path = test_ir_path + 'roadscene' + '_' + str(index) + '.png'
			# visible_path = test_vis_path + 'roadscene' + '_' + str(index) + '.png'
			# infrared_path = test_ir_path + 'video' + '_' + str(index) + '.png'
			# visible_path = test_vis_path + 'video' + '_' +str(index) + '.png'
			run_demo(model1, model2, infrared_path, visible_path, output_path, index, f_type)
	# end = time.time()
	# print("consumption time of generating:%s " % (end - begin))
	print('Done......')



if __name__ == '__main__':
	main()
