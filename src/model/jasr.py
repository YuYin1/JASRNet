from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from utility import get_parameters, weights_init_cpm, find_tensor_peak_batch

import numpy as np
import imageio
import cv2

import matplotlib
from matplotlib import pyplot as plt

def make_model(args, parent=False):
    net = JASR(args)
    net.apply(weights_init_cpm)

    return net


class JASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(JASR, self).__init__()

        nParts = args.nParts + 1
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.sub_mean_cpm = common.MeanShift(args.rgb_range)
        self.num_stages = 3
        self.argmax = 4

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.long_connection = nn.Sequential(conv(n_feats, n_feats, kernel_size),nn.MaxPool2d(kernel_size=2, stride=2),
                                            conv(n_feats, n_feats, kernel_size),nn.MaxPool2d(kernel_size=2, stride=2),
                                            conv(n_feats, n_feats, kernel_size),nn.MaxPool2d(kernel_size=2, stride=2))

        # define encoder
        self.encoder_a = nn.Sequential(
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True),
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True) 
                        )
        self.encoder_a_downsample = nn.Sequential(
                        conv(n_feats, n_feats, kernel_size, stride=2))

        self.encoder_b1 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True),
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True),
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True)) 
        self.encoder_b1_downsample = nn.Sequential(
                        conv(n_feats, n_feats, kernel_size, stride=2))
        self.encoder_b2 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.encoder_c1 = nn.Sequential(
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True), 
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True), 
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True), 
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True))
        self.encoder_c1_downsample = nn.Sequential(
                        conv(n_feats, n_feats, kernel_size, stride=2))
        self.encoder_c2 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True), 
                        common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1), nn.ReLU(inplace=True))
        
        # self.concat_layer = nn.Sequential(conv(n_feats*2, n_feats, kernel_size))
        # define shared features
        m_features = []
        for _ in range(n_resblocks):
            m_features.append(common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1))
            m_features.append(nn.ReLU(inplace=True))

        # ---- SR ---- #
        # define sr_feature module
        m_sr_feature = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(2)
        ]
        m_sr_feature.append(conv(n_feats, n_feats, kernel_size))

        # define sr_tail module
        m_sr_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]


        # ---- CPM ---- #
        m_CPM_feature = []
        for _ in range(2):
            m_CPM_feature.append(common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1))
            m_CPM_feature.append(nn.ReLU(inplace=True))
        m_CPM_feature.append(conv(n_feats, n_feats, kernel_size))

        # define cpm stage module
        stage1 = nn.Sequential(
                common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1),nn.ReLU(inplace=True),
                common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1),nn.ReLU(inplace=True),
                conv(n_feats, nParts, kernel_size)
                )
        stages = [stage1]
        for i in range(1, self.num_stages):
            stagex = nn.Sequential(
                conv(n_feats+nParts, n_feats, kernel_size),nn.ReLU(inplace=True),
                common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1),nn.ReLU(inplace=True),
                common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1),nn.ReLU(inplace=True),
                common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1),nn.ReLU(inplace=True),
                conv(n_feats, nParts, kernel_size)
                )
            stages.append( stagex )
        
        
        self.head = nn.Sequential(*m_head)
        self.features = nn.Sequential(*m_features)
        self.sr_feature = nn.Sequential(*m_sr_feature)
        self.sr_tail = nn.Sequential(*m_sr_tail)
        
        self.CPM_feature = nn.Sequential(*m_CPM_feature)
        self.stages = nn.ModuleList(stages)

    def forward(self, lr):
        assert lr.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(lr.size())
        batch_size, feature_dim = lr.size(0), lr.size(1)
        batch_cpms = []

        # SR feature extraction
        x = self.sub_mean(lr)
        x = self.head(x)  

        a = self.encoder_a(x) 

        b = self.encoder_b1(a) 
        a = self.encoder_a_downsample(a) + b
        a = self.encoder_b1_downsample(a) 
    
        c = self.encoder_b2(b) 
        c = self.encoder_c1(c)
        a = a + c

        a = self.encoder_c1_downsample(a) 
        c = self.encoder_c2(c)

        feat = self.features(c)
        feat = feat + a


        # SR upsample
        sr_feature = self.sr_feature(feat)
        sr_feature = sr_feature + self.long_connection(x) 

        sr = self.sr_tail(sr_feature)
        sr = self.add_mean(sr)

        # CPM 
        xfeature = self.CPM_feature(feat)
        # xfeature = xfeature + x 

        for i in range(self.num_stages):
            if i == 0: cpm = self.stages[i]( xfeature )
            else:      cpm = self.stages[i]( torch.cat([xfeature, batch_cpms[i-1]], 1) )
            batch_cpms.append( cpm )

        return sr,  batch_cpms


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name[:7] == "module.":
                name = name[7:]
            if name in own_state:
                # print("yes\n")
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)

    
    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [ {'params': get_parameters(self.features,   bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.features,   bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.CPM_feature, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                  ]

        for stage in self.stages:
            params_dict.append( {'params': get_parameters(stage, bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay} )
            params_dict.append( {'params': get_parameters(stage, bias=True ), 'lr': base_lr*8, 'weight_decay': 0} )

        return params_dict


