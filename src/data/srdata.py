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

class SRData(data.Dataset):
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

        list_hr, list_lr, list_anno = self._scan()

        if args.ext.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr, list_lr = self._scan()
            self.images_hr = self._check_and_load(
                args.ext, list_hr, self._name_hrbin()
            )
            self.images_lr = [
                self._check_and_load(args.ext, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]
        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr
            elif args.ext.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                os.makedirs(
                    self.dir_anno.replace(self.apath, path_bin),
                    exist_ok=True
                )
                for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        exist_ok=True
                    )
                
                self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                self.images_anno = []
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(
                        args.ext, [h], b, verbose=True, load=False
                    )

                for i, ll in enumerate(list_lr):
                    for l in ll:
                        b = l.replace(self.apath, path_bin)
                        b = b.replace(self.ext[1], '.pt')
                        self.images_lr[i].append(b)
                        self._check_and_load(
                            args.ext, [l], b,  verbose=True, load=False
                        )

                for an in list_anno:
                    b = an.replace(self.apath, path_bin)
                    b = b.replace(self.ext[2], '_anno.pt')
                    self.images_anno.append(b)
                    self._anno_check_and_load(
                        args.ext, [an], b, verbose=True, load=False
                    )

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        names_anno = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_anno.append(os.path.join(self.dir_anno, '{}{}'.format(filename, self.ext[2])))
            for si, s in enumerate(self.scale):
                if self.args.data_train[0] == 'AFLW'  or self.args.data_test[0] == 'AFLW':
                    names_lr[si].append(os.path.join(
                        self.dir_lr, 'X{}/{}{}'.format(
                            s, filename, self.ext[1]
                        )
                    ))
                else:
                    names_lr[si].append(os.path.join(
                        self.dir_lr, 'X{}/{}x{}{}'.format(
                            s, filename, s, self.ext[1]
                        )
                    ))

        return names_hr, names_lr, names_anno

    def _set_filesystem(self, dir_data):
        if hasattr(self, 'specificDataName'):
            self.apath = os.path.join(dir_data, self.name, self.specificDataName)
        else:
            self.apath = os.path.join(dir_data, self.name)
        # self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.dir_anno = os.path.join(self.apath, 'anno')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.jpg', '.jpg', '.pts')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f: pickle.dump(b, _f)

            return b

    def _anno_check_and_load(self, ext, l, f, verbose=True, load=True):
        # generate heatmaps, then write to binary files
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            for _l in l:
                if self.args.data_train[0] == 'AFLW'  or self.args.data_test[0] == 'AFLW':
                    pts, _ = common.anno_parser_v1(_l, self.nParts)
                else:
                    pts, _ = common.anno_parser(_l, self.nParts)
                b = {
                    'name': os.path.splitext(os.path.basename(_l))[0],
                    'pts': pts
                    # 'face_sz': common.faceSZ_from_pts(pts)
                    # 'inter': np.sqrt(torch.sum((iterable[36]-iterable[45]) ** 2).float())
                } 
            with open(f, 'wb') as _f: pickle.dump(b, _f)

            return b

    def __getitem__(self, idx):
        lr, hr, filename, pts, facebb = self._load_file(idx)

        lr, hr, pts = self.get_patch(lr, hr, pts)
        
        # generate heatmaps and masks
        height, width = hr.shape[0], hr.shape[1]
        pts = common.apply_bound(pts, width, height)

        heatmaps, mask = common.generate_label_map(pts, height//self.downsample, width//self.downsample, self.sigma, self.downsample, self.heatmap_type) # H*W*C
        # heatmaps, mask = common.generate_label_map(pts, height//2, width//2, self.sigma, 2, self.heatmap_type) # H*W*C
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
        f_lr = self.images_lr[self.idx_scale][idx]
        f_anno = self.images_anno[idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
            pt = f_anno['pts']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
                pt = load_txt_file(f_anno)
            elif self.args.ext.find('sep') >= 0:

                with open(f_hr, 'rb') as _f: hr = pickle.load(_f)[0]['image']
                with open(f_lr, 'rb') as _f: lr = pickle.load(_f)[0]['image']
                with open(f_anno, 'rb') as _f: pt = pickle.load(_f)['pts']
            if self.name == "AFLW":
                facebb = np.loadtxt(os.path.join(self.dir_hr, "{}_bb.txt".format(filename)), delimiter=' ')
            else:
                facebb = np.array([ pt[0,:].min(), pt[1,:].min(), pt[0,:].max(), pt[1,:].max() ])
               
        return lr, hr, filename, pt, facebb

    def get_patch_no(self, lr, hr, pts):
        scale = self.scale[self.idx_scale]
        
        if self.train:
            if not self.args.no_augment: lr, hr, pts = common.augment(lr, hr, pts)

        return lr, hr, pts

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

