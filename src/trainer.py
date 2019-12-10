import os
import math
from decimal import Decimal

import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from loss import cpm
import cv2
import numpy as np
import time
import statistics

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.optimizer.schedule()
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log('=> Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / (1024. * 1024)))
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, lr_large, heatmaps, mask, pts, _, _, idx_scale) in enumerate(self.loader_train):
            lr, hr, heatmaps, lr_large, mask, pts = self.prepare(lr, hr, heatmaps, lr_large, mask, pts)
            timer_data.hold()
            timer_model.tic()
            
            #################### start training ####################
            self.optimizer.zero_grad()

            ## both
            sr, batch_heatmaps = self.model(lr_large, idx_scale)      

            loss = self.loss(sr=sr, hr=hr, output=batch_heatmaps, target=heatmaps, mask=mask)
            
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )

            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), 3)
        )
        self.model.eval()

        timer_test = utility.timer()
        t_time = []
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, lr_large, heatmaps, mask, pts, filename, facebb, _ in tqdm(d, ncols=80):
                    lr, hr, lr_large = self.prepare(lr, hr, lr_large)
                    heatmaps = heatmaps.cuda(non_blocking=True)

                    #################### start testing #################### 
                    sr, batch_heatmaps = self.model(lr_large, idx_scale)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    self.ckp.log[-1, idx_data, 0] += utility.calc_nme(self.args, pts, batch_heatmaps, mask, hr, facebb, filename, sr)

                    save_list = [sr]
                    psnr, ssim = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d, facebb=facebb)
                    self.ckp.log[-1, idx_data, 1] += psnr
                    self.ckp.log[-1, idx_data, 2] += ssim

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                
                # face alignment
                self.ckp.log[-1, idx_data, 0] /= len(d)
                best = self.ckp.log.min(0)
                
                self.ckp.write_log(
                    '[{} x{}]\tNME: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, 0],
                        best[0][idx_data, 0],
                        best[1][idx_data, 0] + 1
                    )
                )

                #sr psnr
                self.ckp.log[-1, idx_data, 1] /= len(d)
                best_sr = self.ckp.log.max(0)
                self.ckp.write_log(
                    '\t\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.ckp.log[-1, idx_data, 1],
                        best_sr[0][idx_data, 1],
                        best_sr[1][idx_data, 1] + 1
                    )
                )
                #sr ssim
                self.ckp.log[-1, idx_data, 2] /= len(d)
                self.ckp.write_log(
                    '\t\tSSIM: {:.3f}'.format(
                        self.ckp.log[-1, idx_data, 2]
                    )
                )


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            # self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
            self.ckp.save(self, epoch, is_best=(best_sr[1][0, 1] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

