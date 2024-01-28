import os.path

import numpy as np
import torch
import glob
from utils.inception import InceptionV3
from pytorch_fid import fid_score

def _compute_statistics_of_path(args, path, model, batch_size, dims, cuda):
    assert isinstance(path, str)
    assert path.endswith('.npz')
    f = np.load(path)
    if 'mean' in f:
        m, s = f['mean'][:], f['cov'][:]
    else:
        m, s = f['mu'][:], f['sigma'][:]
    f.close()

    return m, s

def get_fid(args, fid_stat, epoch, gen_net, num_img, gen_batch_size, val_batch_size, writer_dict=None, cls_idx=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_net.eval()
    with torch.no_grad():
        # eval mode
        gen_net.eval()

#         eval_iter = num_img // gen_batch_size
#         img_list = []
#         for _ in tqdm(range(eval_iter), desc='sample images'):
#             z = torch.cuda.FloatTensor(np.random.normal(0, 1, (gen_batch_size, args.latent_dim)))

#             # Generate a batch of images
#             if args.n_classes > 0:
#                 if cls_idx is not None:
#                     label = torch.ones(z.shape[0]) * cls_idx
#                     label = label.type(torch.cuda.LongTensor)
#                 else:
#                     label = torch.randint(low=0, high=args.n_classes, size=(z.shape[0],), device='cuda')
#                 gen_imgs = gen_net(z, epoch)
#             else:
#                 gen_imgs = gen_net(z, epoch)
#             if isinstance(gen_imgs, tuple):
#                 gen_imgs = gen_imgs[0]
#             img_list += [gen_imgs]

#         img_list = torch.cat(img_list, 0)
        #fid_score = calculate_fid_given_paths_torch(args, gen_net, fid_stat, gen_batch_size=gen_batch_size, batch_size=val_batch_size)

        inception = InceptionV3()
        model = inception.to(device)

        data_path = args.path_helper['sample_path']
        img_list = glob.glob(os.path.join(data_path, '*.png'))

        print("%d generated images found and loaded" % len(img_list))

        m1, s1 = _compute_statistics_of_path(args, fid_stat, model, batch_size=val_batch_size, dims=2048, cuda=True)
        print("val batch size: ", val_batch_size)
        m2, s2 = fid_score.calculate_activation_statistics(img_list, model, batch_size=val_batch_size, device=device)
        fid_s = fid_score.calculate_frechet_distance(m1, s1, m2, s2)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('FID_score', fid_s, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return fid_s