import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import cv2
from sklearn.metrics import auc
import math


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from data import common
from skimage.measure import compare_psnr, compare_ssim

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_psnr_{}.pdf'.format(d)))
            plt.close(fig)

    def plot_nme(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'Face Alignment on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('NME')
            plt.grid(True)
            plt.savefig(self.get_path('test_nme_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            # for v in save_list:
            #     normalized = v[0].mul(255 / self.args.rgb_range)
            #     tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

            #     self.queue.put((filename, tensor_cpu))

            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def convert_rgb_to_y(tensor):
    image = tensor[0].cpu().numpy().transpose(1,2,0)#.detach()

    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    #xform = np.array([[65.481, 128.553, 24.966]])
    #y_image = image.dot(xform.T) + 16.0

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image

def calc_psnr(sr, hr, scale, rgb_range, dataset=None, facebb=[]):
    # Y channel
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    shave = scale
    facebb = facebb[0].numpy()
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    _, _, w, h = hr.size()
    x1 = max(int(facebb[0]), shave)
    x2 = min(int(facebb[2]), w-shave)
    y1 = max(int(facebb[1]), shave)
    y2 =  min(int(facebb[3]), h-shave)


    image1 = convert_rgb_to_y(sr)
    image2 = convert_rgb_to_y(hr)
    image1 = image1[y1:y2, x1:x2, :]
    image2 = image2[y1:y2, x1:x2, :]
    psnr = compare_psnr(image1, image2, data_range=rgb_range)
    ssim = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                        sigma=1.5, data_range=rgb_range)

    return psnr, ssim

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    # trainable = target.model.specify_parameter(args.lr, args.weight_decay)
    trainable = filter(lambda x: x.requires_grad, target.parameters())


    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
        kwargs_optimizer['nesterov'] = True
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    print(optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def get_parameters(model, bias):
  for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      if bias:
        yield m.bias
      else:
        yield m.weight
    elif isinstance(m, nn.BatchNorm2d):
      if bias:
        yield m.bias
      else:
        yield m.weight

def weights_init_cpm(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def resize_bi(lr, scale, interp='INTER_CUBIC'):
    im = lr.cpu().numpy()[0]
    im = np.transpose(im, (1, 2, 0)) 
    height, width = im.shape[:2]

    im_bi = cv2.resize(im, (width*scale, height*scale), interpolation=cv2.INTER_CUBIC)

    im_bi_tensor = np.transpose(im_bi, (2, 0, 1))
    im_bi_tensor = np.expand_dims(im_bi_tensor, axis=0)

    return torch.from_numpy(im_bi_tensor).cuda()

def find_tensor_peak_batch(heatmap, downsample, threshold = 0.000001):
    radius = 4
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()
  
    def normalize(x, L):
        return -1. + 2. * x.data / (L-1)
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    #affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
    affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
    affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
    affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius+1).to(heatmap).view(1, 1, radius*2+1)
    Y = torch.arange(-radius, radius+1).to(heatmap).view(1, radius*2+1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
    x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
    y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h
     
    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y],1), score

def evaluate_normalized_mean_error(predictions, groundtruth, facebb=None):
  ## compute total average normlized mean error
    # if extra_faces is not None: assert len(extra_faces) == len(predictions), 'The length of extra_faces is not right {} vs {}'.format( len(extra_faces), len(predictions) )
    # num_images = len(predictions)
    # for i in range(num_images):
        # c, g = predictions[i], groundtruth[i]
    # error_per_image = np.zeros((num_images,1))
    num_images = 1
    num_points = predictions.shape[1]
    error_per_image = np.zeros((1))
    
    for i in range(num_images):
        detected_points = predictions
        ground_truth_points = groundtruth
        if num_points == 68:
            interocular_distance = np.linalg.norm(ground_truth_points[:2, 36] - ground_truth_points[:2, 45])
            assert bool(ground_truth_points[2,36]) and bool(ground_truth_points[2,45])
        elif num_points == 51 or num_points == 49:
            interocular_distance = np.linalg.norm(ground_truth_points[:2, 19] - ground_truth_points[:2, 28])
            assert bool(ground_truth_points[2,19]) and bool(ground_truth_points[2,28])
        elif num_points == 19:
            W = facebb[2] - facebb[0]
            H = facebb[3] - facebb[1]
            interocular_distance = np.sqrt(W * H)# common.faceSZ_from_pts(groundtruth) #
        elif num_points == 194:
            interocular_distance = common.faceSZ_from_pts(groundtruth)
        else:
            raise Exception('----> Unknown number of points : {}'.format(num_points))
        dis_sum, pts_sum = 0, 0
        for j in range(num_points):
            if bool(ground_truth_points[2, j]):
                dis_sum = dis_sum + np.linalg.norm(detected_points[:2, j] - ground_truth_points[:2, j])
                pts_sum = pts_sum + 1

        error_per_image = dis_sum / (pts_sum*interocular_distance)

    # normalise_mean_error = error_per_image.mean()
    normalise_mean_error = error_per_image
    # calculate the auc for 0.07
    max_threshold = 0.07
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
    area_under_curve07 = auc(threshold, accuracys) / max_threshold
    # calculate the auc for 0.08
    max_threshold = 0.08
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
    area_under_curve08 = auc(threshold, accuracys) / max_threshold
  
    accuracy_under_007 = np.sum(error_per_image<0.07) * 100. / error_per_image.size
    accuracy_under_008 = np.sum(error_per_image<0.08) * 100. / error_per_image.size

    # print('Compute NME and AUC for {:} images with {:} points :: [(nms): mean={:.3f}, std={:.3f}], auc@0.07={:.3f}, auc@0.08-{:.3f}, acc@0.07={:.3f}, acc@0.08={:.3f}'.format(num_images, num_points, normalise_mean_error*100, error_per_image.std()*100, area_under_curve07*100, area_under_curve08*100, accuracy_under_007, accuracy_under_008))

    for_pck_curve = []
    for x in range(0, 3501, 1):
        error_bar = x * 0.0001
        accuracy = np.sum(error_per_image < error_bar) * 1.0 / error_per_image.size
        for_pck_curve.append((error_bar, accuracy))
  
    return normalise_mean_error, accuracy_under_008, for_pck_curve

def calc_nme(args, pts, batch_heatmaps, mask, hr_np, facebb, filename, sr):    
    argmax = 4
    downsample = hr_np.shape[-1]/batch_heatmaps[0].size()[-1] #args.scale[0]
    batch_size = 1
    # The location of the current batch
    batch_locs, batch_scos = [], []
    for ibatch in range(batch_size):
        batch_location, batch_score = find_tensor_peak_batch(batch_heatmaps[-1][ibatch], downsample)
        batch_locs.append( batch_location )
        batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

    # np_batch_locs: (1, 69, 2)
    np_batch_locs, np_batch_scos = batch_locs.detach().cpu().numpy(), batch_scos.detach().cpu().numpy()    

    for i in range(len(np_batch_locs)):
        locations = np_batch_locs[ibatch,:-1,:]
        scores = np.expand_dims(np_batch_scos[ibatch,:-1], -1)
        
        prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)
        groundtruth = pts[i].numpy()

        facebb = facebb[0].numpy()
        nme, accuracy_under_008, _ = evaluate_normalized_mean_error(prediction, groundtruth, facebb)
    return nme*100

                



