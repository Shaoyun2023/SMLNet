import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified, SPDIncreaseDim
import fusion_strategy
from args_fusion import args
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from torch.nn.functional import unfold
from einops.layers.torch import Rearrange
from einops import rearrange
from timm.models.layers import DropPath
# from t2t_vit import Channel, Spatial
from torch import einsum
import math
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import os



def add_gaussian_noise(feature, std=0.1):
    noise = torch.randn_like(feature) * std  # 生成相同形状的高斯噪声
    noisy_feature = feature + noise
    return noisy_feature

def add_salt_pepper_noise(feature, prob=0.05):

    mask = torch.rand_like(feature)
    noisy_feature = feature.clone()
    noisy_feature[mask < prob/2] = 0    # 椒噪声（黑点）
    noisy_feature[mask > 1-prob/2] = 1  # 盐噪声（白点）
    return noisy_feature



def add_random_occlusion(img, occlusion_size=(64, 64), occlusion_value=0):

    h, w = img.shape[:2]
    occ_h, occ_w = occlusion_size

    # 随机选择遮挡的左上角坐标
    x = np.random.randint(0, w - occ_w)
    y = np.random.randint(0, h - occ_h)

    # 添加遮挡
    if len(img.shape) == 3:  # 彩色图像
        img[y:y + occ_h, x:x + occ_w, :] = occlusion_value
    else:  # 灰度图像
        img[y:y + occ_h, x:x + occ_w] = occlusion_value

    return img


def spd_reduction(matrix, target_dim):
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # 取前 target_dim 个主成分的投影矩阵
    projection_matrix = eigenvectors[:, :target_dim]  # 形状 (2178, 1089)
    # 降维：X_reduced = X @ projection_matrix
    reduced_matrix = matrix @ projection_matrix  # (2178, 2178) @ (2178, 1089) -> (2178, 1089)
    # 由于协方差矩阵是对称的，可以进一步压缩到 (1089, 1089)
    return reduced_matrix[:target_dim, :]  # 取前 1089 行


def laplacian_filter1(tensor):
    # 将输入的 PyTorch 张量转换为 numpy 数组
    np_array = tensor.numpy()

    # 初始化一个和输入张量同形状的空 numpy 数组，用于存储滤波后的结果
    filtered_array = np.zeros_like(np_array)

    # 遍历张量的每一个通道进行Laplacian滤波
    for i in range(np_array.shape[0]):
        # filtered_array[i, 0, :, :] = cv2.Laplacian(np_array[i, 0, :, :], cv2.CV_64F)
        filtered_array[i, 0, :, :] = cv2.Canny(np_array[i, 0, :, :], cv2.CV_64F)

    # 将滤波后的 numpy 数组转回为 PyTorch 张量
    filtered_tensor = torch.from_numpy(filtered_array)
    return filtered_tensor

def gaussian_filter1(tensor, kernel_size=5, sigma=1):
    np_array = tensor.numpy()

    filtered_array = np.zeros_like(np_array)

    for i in range(np_array.shape[0]):
        # convert to 2D
        image_2d = np_array[i, 0, :, :]
        filtered_array[i, 0, :, :] = cv2.GaussianBlur(image_2d, (kernel_size, kernel_size), sigma)

    filtered_tensor = torch.from_numpy(filtered_array)

    return filtered_tensor

def mean_filter1(tensor, kernel_size=5):
    # 将输入的 PyTorch 张量转换为 numpy 数组
    np_array = tensor.numpy()

    # 初始化一个和输入张量同形状的空 numpy 数组，用于存储滤波后的结果
    filtered_array = np.zeros_like(np_array)

    # 遍历张量的每一个通道进行均值滤波
    for i in range(np_array.shape[0]):
        filtered_array[i, 0, :, :] = cv2.blur(np_array[i, 0, :, :], (kernel_size, kernel_size))

    # 将滤波后的 numpy 数组转回为 PyTorch 张量
    filtered_tensor = torch.from_numpy(filtered_array)
    return filtered_tensor



def get_patches(input, kernel_size, stride):

    patches = F.unfold(input, kernel_size, stride=stride,padding=8)
    patches = patches.transpose(1, 2).contiguous().view(patches.size(0)*patches.size(-1), -1)

    return patches

# def fold_with_overlap_average(patches):
#     patches = patches.transpose(1, 2).contiguous()
#     # 对 patches 做 fold 操作
#     output = F.fold(patches, (256,256), kernel_size=16, stride=14)
#     torch.set_printoptions(threshold=float('inf'))
#
#     print(output)
#
#     # print(output.shape)
#
#     image = output.squeeze().detach().cpu().numpy()
#
#     image = (image - image.min()) / (image.max() - image.min())
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.show()
#
#
#     return output
def average(num1, num2):
    return (num1 + num2) / 2

def fold_with_overlap_average(patches):
    patches = patches.transpose(1, 2).contiguous()
    # 对 patches 做 fold 操作
    output = F.fold(patches, (256,256), kernel_size=16, stride=8,padding=8)



    # torch.set_printoptions(threshold=float('inf'))
    # # patches = patches.view(1,-1,256).contiguous()
    # patches = patches.transpose(1, 2).contiguous()
    # patches = patches.view(-1).contiguous()


    # for k in range(17):                               #行遍历18次
    #     for j in range(18):                           #列遍历16次
    #         for i in range(16):                       #上下两个patch之间的距离
    #             output[0, 0, i + j * 14, 14+k*14] = average(patches[14 + 256 * j * 18 + 256 * k + i * 16],patches[256 + 256 * j * 18 + 256 * k + i * 16])
    #             output[0, 0, i + j * 14, 15+k*14] = average(patches[15 + 256 * j * 18 + 256 * k + i * 16], patches[257 + 256 * j * 18  + 256 * k + i * 16])
    #
    # for k in range(17):
    #     for j in range(18):
    #         for i in range(16):
    #             output[0, 0, 14+k*14, i + 14 * j] = average(patches[224 + i + 256 * j + 18 * 256 * k], patches[256 * 18 + i + 256 * j + 18 * 256 * k])
    #             output[0, 0, 15+k*14, i + 14 * j] = average(patches[224 + 16 + i + 256 * j + 18 * 256 * k], patches[256 * 18 + 16 + i + 256 * j + 18 * 256 * k])





    # image = output.squeeze().detach().cpu().numpy()
    #
    # image = (image - image.min()) / (image.max() - image.min())
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()


    return output




def overlapping_patches(input_tensor):
    # 初始化结果矩阵
    result_matrix = torch.zeros((254, 254))

    # 将tensor形状改为324x16x16
    reshaped_tensor = input_tensor.view(-1, 16, 16)

    row_position = 0
    for i in range(0, reshaped_tensor.shape[0] // 18):  # 每18个patch一行，循环次数为行数
        col_position = 0
        for j in range(18):  # 每行有18个patch
            patch = reshaped_tensor[i * 18 + j]  # 获取当前的patch

            # 将patch填入结果矩阵的对应位置，并处理重叠区域
            if row_position != 0:  # 不是第一行的时候，需要处理上方的重叠区域
                result_matrix[row_position:row_position + 16, col_position:col_position + 16] = patch
                result_matrix[row_position:row_position + 2, col_position:col_position + 16] = \
                    (result_matrix[row_position:row_position + 2, col_position:col_position + 16] +
                     result_matrix[row_position - 2:row_position, col_position:col_position + 16]) / 2
            elif col_position != 0:  # 不是第一列的时候，需要处理左方的重叠区域
                result_matrix[row_position:row_position + 16, col_position:col_position + 16] = patch
                result_matrix[row_position:row_position + 16, col_position:col_position + 2] = \
                    (result_matrix[row_position:row_position + 16, col_position:col_position + 2] +
                     result_matrix[row_position:row_position + 16, col_position - 2:col_position]) / 2
            else:  # 第一行第一列（左上角）的patch直接放入结果矩阵
                result_matrix[row_position:row_position + 16, col_position:col_position + 16] = patch

            col_position += 14  # 每次右移14个单位，为了处理下一个patch

        row_position += 14  # 每处理完一行后，下移14个单位，为了处理下一行

    return result_matrix


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out



class PatchFlattener1(nn.Module):
    def __init__(self, patch_size):
        super(PatchFlattener1, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Invalid patch size"
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        # Rearrange patches into columns
        x = x.view(batch_size, channels, patch_height, self.patch_size, patch_width, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, num_patches, -1)
        return x

class PatchFlattener2(nn.Module):
    def __init__(self, patch_size):
        super(PatchFlattener2, self).__init__()
        self.patch_size = patch_size
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Invalid patch size"
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        # Rearrange patches into columns
        x = x.view(batch_size, channels, patch_height, self.patch_size, patch_width, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, num_patches, -1)
        return x


def overlap_patch(input_img, patch_size=16, overlap=2, filter_func=None):
    """
    :param input_img: 输入图片，np.array，shape = [1,1,256,256]
    :param patch_size: 分块的尺寸，默认为16
    :param overlap: 块与块之间的重叠部分，默认为4
    :param filter_func: 用于特征提取的滤波函数
    """
    input_img = np.squeeze(input_img)  # 消除单维度条目
    width, height = input_img.shape  # 图片的宽度和高度

    # 创建一个空的数组来存储处理过的图片
    processed_img = np.zeros_like(input_img.cpu().detach().numpy())

    step_size = patch_size - overlap  # 步长 = 分块尺寸 - 重叠部分


    for i in range(0, width, step_size):
        for j in range(0, height, step_size):
            # 分块
            patch = input_img[i: i + patch_size, j: j + patch_size]

            if filter_func:
                # 特征提取
                patch = filter_func(patch)
            # patch = patch.cpu().detach().numpy()
            # 恢复原位置
            processed_img[i: i + patch_size, j: j + patch_size] = patch

    return processed_img.reshape([1, 1, width, height])




class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out




class SPDconv(nn.Module):
    def __init__(self, in_chans=1, out_chans=1):
        super(SPDconv, self).__init__()

        # self.trans9 = SPDTransform(2178, 1634)
        # self.trans10 = SPDTransform(1634, 1089)

        # self.trans9 = SPDTransform(2178, 1815)
        # self.trans10 = SPDTransform(1815, 1452)
        # self.trans11 = SPDTransform(1452,1089)

        self.trans9 = SPDTransform(2178,1089)

        # self.trans10 = SPDTransform(1089,500)
        # self.trans11 = SPDTransform(500,200)


        self.rect1 = SPDRectified()
        # self.rectconv = SPDRectified()
        # self.tangent = SPDTangentSpace(32)  # 映射回欧式空间
        self.tangentconv = SPDTangentSpace(1089)

        self.averagepool = nn.AvgPool2d(kernel_size=16, stride=16, padding=0)


        self.patch_flattener1 = PatchFlattener1(16)
        self.patch_flattener2 = PatchFlattener2(16)

        # self.overleap1 = PatchFlattener(16)
        # self.overleap2 = PatchReconstructor(16,1,256)
        # self.overlap = OverlappingPatchFlattener(patch_size=16,overlap=8)
        self.heatmap_counter = 1
        self.save_counter = 1

    def spdconv(self, ir, vi):  # input[1,16,224,224]

        ir = ir.double()
        vi = vi.double()

        oir = ir
        ovi = vi

        oir = oir.to(args.device)
        ovi = ovi.to(args.device)

        oir = get_patches(oir,16,8)                                     #[324,256]
        o11 = oir
        oir = oir.cpu().detach().numpy()

        ovi = get_patches(ovi,16,8)
        o22 = ovi
        ovi = ovi.cpu().detach().numpy()

        cov = np.vstack((oir, ovi))
        cov = np.cov(cov, rowvar=True)

        cov = torch.from_numpy(cov)



        U, s, VT = torch.svd(cov)

        Sigma = np.zeros(np.shape(cov))
        Sigma[:len(s), :len(s)] = np.diag(s)


        trace = np.trace(Sigma)
        B = np.zeros_like(Sigma)
        np.fill_diagonal(B, trace)

        Sigma = Sigma + B * 0.001
        cov = U @ Sigma @ U.T


        x = torch.unsqueeze(cov, dim=0)


        spd1 = x
        # x = self.trans7(x)
        # x = self.rect1(x)

        spd2 = x


        x = self.trans9(x)
        x = self.rect1(x)

        # x = self.trans10(x)
        # x = self.rect1(x)
        #
        # x = self.trans11(x)
        # x = self.rect1(x)

        spd3 = x
        x = self.tangentconv(x)


        x = x.unsqueeze(0)
        x = x.double()


        # for i in range(1089):
        #     x[0, 0 ,i, i] = 8
        weight = x

        x = x.to(args.device)



        # x = x.squeeze(0).squeeze(0)
        # rows, cols = x.size()
        # row_half = rows // 2
        # col_half = cols // 2
        # # 第一象限
        # x[:row_half, col_half:] = -x[:row_half, col_half:]
        # # 第三象限
        # x[row_half:, :col_half] = -x[row_half:, :col_half]
        # # for i in range(1089):
        # #     x[i, i] =0
        # x = x.unsqueeze(0).unsqueeze(0)




        out1 =  torch.matmul(x, o11)
        out2 =  torch.matmul(x, o22)

        out1 = out1.squeeze(0)
        out2 = out2.squeeze(0)

        out1 = fold_with_overlap_average(out1)
        out2 = fold_with_overlap_average(out2)


        out1 = out1.float()
        out2 = out2.float()

        result1 = out1
        result2 = out2


        return [result1,result2, spd1, spd2, spd3, weight]

    def spdconvx(self, x):  # input[1,16,224,224]
        x = x.double()

        s = self.patch_flattener1(x)
        s = torch.squeeze(s, dim=0)
        s1 = s
        s = s.cpu().detach().numpy()

        cov = np.cov(s, rowvar=True)
        # cov = np.cov(sir, svi, rowvar=True)[:256, 256:]

        cov = torch.from_numpy(cov)

        x = torch.matmul(cov.to(args.device),s1.to(args.device))

        # weight_min1 = torch.min(x)
        # weight_max1 = torch.max(x)
        # x = (x - weight_min1) / (weight_max1 - weight_min1)

        # image = x.squeeze().detach().cpu().numpy()
        #
        # image = (image - image.min()) / (image.max() - image.min())
        # plt.imshow(image, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # x = cov.numpy()
        return [x]

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        b, c, h, w = X.shape
        if c == 1:
            X = X.repeat(1, 3, 1, 1)

        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        # [relu1_2, relu2_2, relu3_3, relu4_3]
        return [relu1_2, relu2_2, relu3_3, relu4_3]


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# NestFuse network - light, no desnse
class SMLNet(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=False):
        super(SMLNet, self).__init__()
        self.deepsupervision = deepsupervision
        # block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1



        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2)
        # self.up_eval = UpsampleReshape_eval()


        self.conve1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());
        self.conve2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());
        self.conve3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());
        self.conve4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());

        self.convd1 = ConvLayer(1, 16, 3, stride)
        self.convd2 = ConvLayer(16, 32, kernel_size, stride)
        self.convd3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.convd4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.convd5 = ConvLayer(64, output_nc, 3, stride)

        self.convd6 = ConvLayer(64, 48, 3, stride)
        self.convd7 = ConvLayer(48, 32, 3, stride)
        self.convd8 = ConvLayer(32, 16, 3, stride)
        self.convd9 = ConvLayer(16, 1, 3, stride, is_last=True)

        # self.convd62 = ConvLayer(128, 64, 3, stride)
        # self.convd72 = ConvLayer(64, 32, 3, stride)
        # self.convd82 = ConvLayer(32, 16, 3, stride)
        # self.convd92 = ConvLayer(16, 1, 3, stride, is_last=True)
        # self.convd = ConvLayer(16, 64, kernel_size, stride)


        self.heatmap_counter = 1

        self.linear1 = nn.Linear(256, 128)
        self.pos_drop = nn.Dropout(p=0.)
        self.linear2 = nn.Linear(128, 256)
        out_channels_def = 16
        in_channels = 1

        nb_filter = [16, 64, 32, 16]
        denseblock = DenseBlock
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)





        # self.convd6 = ConvLayer(128, output_nc, 3, stride)

        # self.sa = SpatialAttention()
        # self.SEblock = SELayer(32)
        # self.convse = ConvLayer(1, 8, 3, stride)
        # self.convsed = ConvLayer(16, 1, 3, stride,is_last=True)
        # self.strans3 = Spatial(size=256, embed_dim=1024 * 2, patch_size=4, channel=64)

        # self.overlap = PatchOverlapping(patch_size=16, stride=8)
        # self.overlap = OverlappingPatchFlattener(patch_size=16, overlap=8)

        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        #
        # self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])



    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def fusion(self, en1, en2, p_type):

        # attention weight
        fusion_function = fusion_strategy.channel_fusion

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        # f1_0 = fusion_function(en1[0], en2[0], p_type)
        # f2_0 = fusion_function(en1[1], en2[1], p_type)
        # f3_0 = fusion_function(en1[2], en2[2], p_type)
        # f4_0 = fusion_function(en1[3], en2[3], p_type)
        return f1_0

        #return [f1_0]

    def encodertrain(self, spdd11, spdd12):
        x1 = self.convd1(spdd11)
        x2 = self.convd1(spdd12)

        x1 = self.convd2(x1)
        x2 = self.convd2(x2)

        x = torch.cat([x1,x2],dim=1)

        x = self.convd6(x)
        x = self.convd7(x)
        x = self.convd8(x)
        x = self.convd9(x)
        # image = x.squeeze().detach().cpu().numpy()
        #
        # # Normalize to [0, 1] range
        # image = (image - image.min()) / (image.max() - image.min())
        #
        # # Create figure without axes
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image, cmap='jet', vmin=0, vmax=1)  # Explicitly set value range
        # plt.axis('off')
        # # plt.show()
        #
        # # Ensure output directory exists
        # output_dir = "results/heatmaps_MSRS"
        # os.makedirs(output_dir, exist_ok=True)
        #
        # # Save with tight layout and proper DPI
        # output_path = os.path.join(output_dir, f"{self.heatmap_counter}.png")
        # plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()  # Important: close the figure to free memory
        #
        # self.heatmap_counter += 1
        return [x]


    def encoderspdtest(self, spdd11, spdd12):
        x1 = self.convd1(spdd11)
        x1 = self.convd2(x1)
        x2 = self.convd1(spdd12)
        x2 = self.convd2(x2)
        x = torch.cat([x1,x2],dim=1)

        x = self.convd6(x)
        x = self.convd7(x)
        x = self.convd8(x)
        x = self.convd9(x)

        # x = torch.cat([x1,x2],dim=1)
        # x = x1+x2
        # x = self.convd1(x)
        # x = self.DB1(x)
        # x_min2 = torch.min(x2)
        # x_max2 = torch.max(x2)
        # x = (x - x_min2) / (x_max2 - x_min2) * 255
        return [x]

    def decoder(self, x):
        # x = self.conv64to48n(x)
        # x = self.conv48to32n(x)
        # x = self.conv32to16n(x)
        # x = self.conv16to1n(x)
        x2 = self.convd2(x)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x = self.convd5(x4)

        # image = torch.sum(x, dim=1, keepdim=True)

        return [x]