import os
import Place365_128_cfg as cfg
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import cv2
from torchvision.transforms import ToTensor, ToPILImage, GaussianBlur
from models_search import Celeba256_gen_mask, Places128_gen

model_path = "mode_place365_deep_epoch10_mask_checkpoint"
args = cfg.parse_args()

# 加载模型
model = Places128_gen.Generator(args=args).cuda()
model = nn.DataParallel(model).cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['avg_gen_state_dict'])
del checkpoint


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image


def Celeba256_mask_infer(img, mask):
    with torch.no_grad():
        img_cv = np.array(img)[:, :, :3]
        img_tensor = (ToTensor()(img_cv) * 2.0 - 1.0).unsqueeze(0)
        mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (img_tensor.shape[0], args.latent_dim))).cuda(args.gpu,
                                                                                                  non_blocking=True)
        z_mask = mask = mask_tensor[0].cuda()
        z_mask = z_mask[0]
        maxPool = nn.AdaptiveAvgPool2d((args.latent_dim, args.latent_dim))
        z_mask = maxPool(z_mask.unsqueeze(0))
        z_mask = z_mask[0]
        z_mask = (1 - z_mask) * args.mask_eps + z_mask * (1 - args.mask_eps)
        # GaussianBlur Soft
        # unloader = ToPILImage()
        # z_mask_pil = unloader(z_mask)
        # z_mask_pil = GaussianBlur(3, 2.0)(z_mask_pil)
        # z_mask = ToTensor()(z_mask_pil).squeeze(0).cuda()

        masked_z = torch.mm(z, z_mask)
        comp_tensor = model(masked_z.cuda(args.gpu), mask.unsqueeze(0).cuda(args.gpu), img_tensor, 0).detach()
        comp_np = postprocess(comp_tensor[0])

        return comp_np


im_path = r"image/"
mas_path = r"./mask/657_mask.jpg"
# len(os.listdir(im_path))
for i in range(len(os.listdir(im_path))):
    print("Deal with No.{} Picture".format(i))
    img = cv2.resize(cv2.imread(os.path.join(im_path, "{}.jpg".format(i))), (128, 128))
    mask = cv2.resize(cv2.imread(mas_path), (128, 128))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    comp_np = Celeba256_mask_infer(img, mask)
    cv2.imwrite(r"result/result{}.jpg".format(i), comp_np)
