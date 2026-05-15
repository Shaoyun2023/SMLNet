import numpy as np
import torch
import torch.nn.functional as F
import imageio
from PIL import Image
from args_fusion import args


def scipy_misc_imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        image = imageio.imread(path)
    else:
        image = imageio.imread(path)

    if height is not None and width is not None:
        image = scipy_misc_imresize(image, [height, width], interp='nearest')

    return image


def _positions(size, patch_size, stride):
    if size <= patch_size:
        return [0]
    pos = list(range(0, size - patch_size, stride))
    if not pos or pos[-1] != size - patch_size:
        pos.append(size - patch_size)
    return pos


def _pad_image(image, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return image
    pad_spec = ((0, pad_h), (0, pad_w))
    if min(image.shape[0], image.shape[1]) > 1:
        return np.pad(image, pad_spec, mode='reflect')
    return np.pad(image, pad_spec, mode='edge')


def _build_weight_map(patch_size, device):
    w = torch.hann_window(patch_size, periodic=False, device=device)
    weight = torch.outer(w, w)
    weight = weight / weight.max().clamp(min=1e-6)
    weight = weight.clamp(min=1e-6)
    return weight


def _patch_to_2d(patch):
    if patch.dim() == 4:
        return patch[0, 0]
    if patch.dim() == 3:
        if patch.shape[0] == 1:
            return patch[0]
        return patch.mean(0)
    return patch


def split_image_to_patches(image, patch_size=256, stride=128):
    h, w = image.shape[:2]
    padded_h = max(h, patch_size)
    padded_w = max(w, patch_size)
    pad_h = padded_h - h
    pad_w = padded_w - w
    padded = _pad_image(image, pad_h, pad_w)

    pos_y = _positions(padded_h, patch_size, stride)
    pos_x = _positions(padded_w, patch_size, stride)

    patches = []
    for y in pos_y:
        for x in pos_x:
            patch = padded[y:y + patch_size, x:x + patch_size]
            patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            patches.append(patch_t)

    meta = {
        "original_size": (h, w),
        "padded_size": (padded_h, padded_w),
        "patch_size": patch_size,
        "stride": stride,
        "positions_y": pos_y,
        "positions_x": pos_x,
    }
    return patches, meta


def reconstruct_from_patches(patches, meta, device=None):
    if not patches:
        raise ValueError("patches is empty")

    device = device or (patches[0].device if torch.is_tensor(patches[0]) else args.device)
    patch_size = meta["patch_size"]
    padded_h, padded_w = meta["padded_size"]
    pos_y = meta["positions_y"]
    pos_x = meta["positions_x"]

    weight = _build_weight_map(patch_size, device)
    output = torch.zeros(1, padded_h, padded_w, device=device)
    weight_sum = torch.zeros(1, padded_h, padded_w, device=device)

    idx = 0
    for y in pos_y:
        for x in pos_x:
            patch = _patch_to_2d(patches[idx]).to(device)
            output[:, y:y + patch_size, x:x + patch_size] += patch * weight
            weight_sum[:, y:y + patch_size, x:x + patch_size] += weight
            idx += 1

    output = output / weight_sum.clamp(min=1e-6)
    h, w = meta["original_size"]
    return output[:, :h, :w]


def get_test_image2(paths, height=None, width=None, flag=False, patch_size=256, stride=128):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    meta = None
    viimage = None
    h = w = 0

    for path in paths:
        image = get_image(path, height, width, flag)
        if height is not None and width is not None:
            image = scipy_misc_imresize(image, [height, width], interp='nearest')

        if image.ndim == 3:
            viimage = image
            image = np.sum(image, axis=2)

        image = image.astype(np.float32)
        h, w = image.shape[:2]
        patches, meta = split_image_to_patches(image, patch_size=patch_size, stride=stride)
        images = patches

    c = len(images)
    return images, h, w, c, viimage, meta


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.reshape([1, img_fusion.shape[0], img_fusion.shape[1]])
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    import cv2
    cv2.imwrite(output_path, img_fusion)
