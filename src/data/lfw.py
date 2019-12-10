import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2

class lfw(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        # landmark parameter
        self.nParts = args.nParts
        self.sigma = 4.0
        self.heatmap_type = 'gaussian'
        self.downsample = args.scale[0]
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        self.images_hr, self.images_lr = self._scan()

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    # Z:\Dataset\lfw\lfw-align-128\Aaron_Eckhart\XX.jpg
    def _scan(self):
        scale = self.scale[0]
        names_hr = []
        names_lr = []
        # all_hr_names = sorted(glob.glob(os.path.join(self.dir_hr, '*.png')))
        all_names = sorted(glob.glob(os.path.join(self.dir_hr, '*')))
        for hr_name in all_names:
            hr_imgs = sorted(glob.glob(os.path.join(hr_name, '*.jpg')))
            for hr_img in hr_imgs:
                names_hr.append(hr_img)
                names_lr.append(hr_img)
        # names_hr = names_hr[self.begin - 1:self.end]
        # names_lr = names_lr[(self.begin - 1) * self.n_seq : self.end * self.n_seq]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name, 'lfw-align-128-sr')

        self.dir_hr = self.apath # os.path.join(self.apath, 'HR')
        self.dir_lr = self.apath # os.path.join(self.apath, 'LR_bicubic')
        # self.dir_anno = os.path.join(self.apath, 'anno')
        # if self.input_large: self.dir_lr += 'L'
        self.ext = ('.jpg', '.jpg')

    def __getitem__(self, idx):
        lr, hr, filename, pts, facebb = self._load_file(idx)

        lr, hr, pts = self.get_patch(lr, hr, pts)
        
        # generate heatmaps and masks
        height, width = hr.shape[0], hr.shape[1]
        pts = common.apply_bound(pts, width, height)

        heatmaps, mask = common.generate_label_map(pts, height//self.downsample, width//self.downsample, self.sigma, self.downsample, self.heatmap_type) # H*W*C
        # cv2.imshow("heatmaps", np.sum(heatmaps[:, :, 0:-1],2))
        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
        mask = torch.from_numpy(mask.transpose((2,0,1))).type(torch.ByteTensor)
        # print(mask.size()) #(69, 1, 1)

        pair = [lr, hr]

        # pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair.append(common.resize_bi(pair[0], self.scale[0], 'INTER_CUBIC'))
        # pair: lr, hr, lr_large
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        
        '''
        # print(pair[0].shape, pair[1].shape, pair[2].shape, pair[3].shape)
        # print(pair_t[0].size(), pair_t[1].size(), pair_t[2].size())
        # (28, 28, 3) (224, 224, 3) (28, 28, 69) (224, 224, 3)
        # print(pair[0].shape, pair[1].shape, np.sum(pair[2],2).shape, len(filename))
        # show pts
        img = hr.copy()
        for i in range(pts.shape[1]):
            cv2.circle(img,(int(pts[0, i]), int(pts[1, i])), 2, (0,255,0), -1)
        cv2.imshow("hr",img)
        
        # show heatmap
        cv2.imshow("lr",pair[0])
        # cv2.imshow("hr",pair[1])
        # cv2.imshow("heatmaps", np.sum(pair[2][:, :, 0:-1],2))
        cv2.imshow("lr_large", pair[2])

        # gt = pts[0].numpy()
        # for i in range(gt.shape[1]):
            # cv2.circle(gt[1],(int(gt[0, i]), int(gt[1, i])), 4, (0,255,0), -1)
            # cv2.circle(pred[1],(int(pred[0, i]), int(pred[1, i])), 5, (0,255,0), -1)
        hr_np = pair_t[1].numpy().transpose(1,2,0)#.detach().cpu()
        cv2.imshow("hr_tensor",hr_np)
        cv2.waitKey(0)
        '''
        # lr, hr,  lr_large, heatmaps,mask
        return pair_t[0], pair_t[1], pair_t[2], heatmaps, mask, pts, filename, facebb

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]


        # filename, _ = os.path.splitext(os.path.basename(f_hr))
        filename = f_hr
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)
        pt = np.zeros([3,68])

        im_bi = cv2.resize(lr, (16, 16), interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(im_bi, (128, 128), interpolation=cv2.INTER_CUBIC)

        facebb = np.array([ 0, 0, 128, 128 ])
               
        return lr, hr, filename, pt, facebb


    def get_patch(self, lr, hr, pts):
        scale = self.scale[self.idx_scale]
        
        if self.train:
            lr, hr, pts = common.get_patch(
                lr, hr, pts,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr, pts = common.augment(lr, hr, pts)

        return lr, hr, pts

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

