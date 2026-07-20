import os
import torch
from torch.autograd import Variable
from net import SMLNet, SPDconv
import utils_robust as utils
from args_fusion import args
import numpy as np
from PIL import Image


def tensor_to_pil(tensor, mode='RGB'):
    """
    Convert torch tensor to PIL image.
    Args:
        tensor: torch.Tensor with shape [C, H, W]
        mode: PIL image mode, e.g. 'RGB', 'L'
    Returns:
        PIL.Image
    """
    tensor = tensor.cpu().detach()

    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [C, H, W], got {tensor.shape}")

    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0

    if tensor.min() < 0 or tensor.max() > 1:
        tensor = tensor.clamp(0, 1)

    if tensor.shape[0] == 3:
        numpy_array = tensor.permute(1, 2, 0).numpy()
    else:
        numpy_array = tensor.squeeze(0).numpy()

    numpy_array = (numpy_array * 255).astype(np.uint8)
    pil_image = Image.fromarray(numpy_array, mode=mode)

    return pil_image


def clamp(value, min=0.0, max=1.0):
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = (out - out.min()) / (out.max() - out.min())
    out = clamp(out)
    return out


def load_model1(path, deepsupervision):
    input_nc = 1
    output_nc = 1
    nb_filter = [16, 64, 32, 16]

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
    return SPD_model


def run_demo(nest_model, SPD_model, infrared_path, visible_path, output_path_root, index, f_type):
    img_ir, h, w, c, viimage_ir, meta_ir = utils.get_test_image2(infrared_path)
    img_vi, h, w, c, viimage_vi, meta_vi = utils.get_test_image2(visible_path)

    meta = meta_vi if meta_vi is not None else meta_ir

    Y_vi = Cb_vi = Cr_vi = None
    if viimage_vi is not None and len(viimage_vi.shape) == 3:
        viimage_tensor = torch.from_numpy(viimage_vi).permute(2, 0, 1)
        if args.cuda:
            viimage_tensor = viimage_tensor.to(args.device)
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(viimage_tensor)

    img_fusion_blocks = []

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
        img_fusion_blocks.append(bbb)

    fused_y = utils.reconstruct_from_patches(img_fusion_blocks, meta, device=args.device)
    img_fusion_list = [fused_y]

    output_count = 0
    for img_fusion in img_fusion_list:
        if Y_vi is None:
            if index < 10:
                file_name = '0' + str(index) + '.png'
            else:
                file_name = str(index) + '.png'
            output_path = output_path_root + file_name
            output_count += 1
            utils.save_image_test(img_fusion, output_path)
            print(f"Saved: {output_path}")
            continue

        Y_fused = img_fusion

        if Y_fused.shape[-2:] != Cb_vi.shape[-2:]:
            Y_fused_resized = torch.nn.functional.interpolate(
                Y_fused.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            Y_fused_resized = Y_fused

        rgb_fused_image = YCrCb2RGB(Y_fused_resized, Cb_vi, Cr_vi)

        if index < 10:
            file_name = '0' + str(index) + '.png'
        else:
            file_name = str(index) + '.png'
        output_path = output_path_root + file_name
        output_count += 1

        pil_image = tensor_to_pil(rgb_fused_image)
        pil_image.save(output_path)
        print(f"Saved: {output_path}")


def main():
    test_path = "images/test_VOT-RGBT/"
    network_type = 'SwinFuse'
    fusion_type = ['l1_mean']
    output_path = 'outputs/VOT-RGBT/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    in_chans = 1
    num_classes = in_chans
    model_path = args.model_default

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[1])
        ssim_weight_str = args.ssim_path[1]
        f_type = fusion_type[0]

        model1 = load_model1(model_path, num_classes)
        model2 = load_model2(model_path, num_classes)

        for i in range(999):
            index = i + 1


            infrared_path = os.path.join(test_path, 'ir', f"{index:02d}.png")
            visible_path = os.path.join(test_path, 'vi', f"{index:02d}.png")

            run_demo(model1, model2, infrared_path, visible_path, output_path, index, f_type)

    print('Done......')


if __name__ == '__main__':
    main()
