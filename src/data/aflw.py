import os
from data import srdata
import numpy as np

class AFLW(srdata.SRData):
    def __init__(self, args, name='AFLW', train=True, benchmark=False):
        self.specificDataName = "AFLW_lr16_hr128"
        # self.specificDataName = "AFLW_lr64_hr256"

        print("Dataset: {}".format(self.specificDataName))

        # test: 0-689
        data_range = [r.split('-') for r in args.data_range.split('/')]
        self.train = train
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        args.nParts = 19
        self.nParts = args.nParts
        super(AFLW, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr, names_anno = super(AFLW, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        names_anno = names_anno[self.begin - 1:self.end]

        return names_hr, names_lr, names_anno

    def _set_filesystem(self, dir_data):
        super(AFLW, self)._set_filesystem(dir_data)
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'AFLW_train_HR')
            self.dir_lr = os.path.join(self.apath, 'AFLW_train_LR_bicubic')
            self.dir_anno = os.path.join(self.apath, 'AFLW_train_HR')
        else:
            self.dir_hr = os.path.join(self.apath, 'AFLW_test_HR')
            self.dir_lr = os.path.join(self.apath, 'AFLW_test_LR_bicubic')
            self.dir_anno = os.path.join(self.apath, 'AFLW_test_HR')
        
        if self.input_large: self.dir_lr += 'L'
    


'''
class AFLW(data.Dataset):
    def __init__(self, args, name='AFLW', train=True, benchmark=False):

    def __init__(self, transform, sigma, downsample, heatmap_type, data_indicator):
        self.reset()
        print ('The general dataset initialization done : {:}'.format(self))

        self.dataset_name = name
        num_pts = 19
        pre_crop_expand = 0.2 
        sigma = 4 
        crop_perturb_max = 30 
        rotate_max = 20 
        scale_prob = 1.0 
        scale_min = 0.9 
        scale_max = 1.1 
        scale_eval = 1 
        heatmap_type = "gaussian"
        crop_height = 128
        crop_width = 128

        if train == True:
            data_lists = "./cache_data/lists/AFLW/train.GTL"

            mean_fill = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
            assert arg_flip == False, 'The flip is : {}, rotate is {}'.format(arg_flip, rotate_max)
            transform  = [transforms.PreCrop(pre_crop_expand)]
            transform += [transforms.TrainScale2WH((crop_width, crop_height))]
            transform += [transforms.AugScale(scale_prob, scale_min, scale_max)]
            #if args.arg_flip:
            #  train_transform += [transforms.AugHorizontalFlip()]
            if rotate_max:
                transform += [transforms.AugRotate(rotate_max)]
            transform += [transforms.AugCrop(crop_width, crop_height, crop_perturb_max, mean_fill)]
            transform += [transforms.GeneSrLrImg(crop_width, crop_height, model_config.downsample)]
            transform += [transforms.ToTensor()]#, normalize
            transform  = transforms.Compose( train_transform )
        else:
            data_lists "./cache_data/lists/AFLW/test.GTL"

            transform  = transforms.Compose([transforms.PreCrop(args.pre_crop_expand), 
                                transforms.TrainScale2WH((args.crop_width, args.crop_height)),  
                                transforms.GeneSrLrImg(args.crop_width, args.crop_height, model_config.downsample),
                                transforms.ToTensor()])

        

        self.load_list(data_lists, self.NUM_PTS, True)


    def __repr__(self):
        return ('{name}(point-num={NUM_PTS}, sigma={sigma}, heatmap_type={heatmap_type}, length={length}, dataset={dataset_name})'.format(name=self.__class__.__name__, **self.__dict__))

    def reset(self, num_pts=-1):
        self.length = 0
        self.NUM_PTS = num_pts
        self.datas = []
        self.labels = []
        self.face_sizes = []
        assert self.dataset_name is not None, 'The dataset name is None'

    def __len__(self):
        assert len(self.datas) == self.length, 'The length is not correct : {}'.format(self.length)
        return self.length

    def append(self, data, label, box, face_size):
        assert osp.isfile(data), 'The image path is not a file : {}'.format(data)
        self.datas.append( data )
        if (label is not None) and (label.lower() != 'none'):
            if isinstance(label, str):
                assert osp.isfile(label), 'The annotation path is not a file : {}'.format(label)
                np_points, _ = anno_parser(label, self.NUM_PTS)
                meta = Point_Meta(self.NUM_PTS, np_points, box, data, self.dataset_name)
            elif isinstance(label, Point_Meta):
                meta = label.copy()
            else:
                raise NameError('Do not know this label : {}'.format(label))
        else:
            meta = Point_Meta(self.NUM_PTS, None, box, data, self.dataset_name)
        self.labels.append( meta )
        self.face_sizes.append( face_size )
        self.length = self.length + 1

    def prepare_input(self, image, box):
        meta = Point_Meta(self.NUM_PTS, None, np.array(box), image, self.dataset_name)
        image = pil_loader( image )
        return self._process_(image, meta, -1), meta

    def load_data(self, datas, labels, boxes, face_sizes, num_pts, reset):
        # each data is a png file name
        # each label is a Point_Meta class or the general pts format file (anno_parser_v1)
        assert isinstance(datas, list), 'The type of the datas is not correct : {}'.format( type(datas) )
        assert isinstance(labels, list) and len(datas) == len(labels), 'The type of the labels is not correct : {}'.format( type(labels) )
        assert isinstance(boxes, list) and len(datas) == len(boxes), 'The type of the boxes is not correct : {}'.format( type(boxes) )
        assert isinstance(face_sizes, list) and len(datas) == len(face_sizes), 'The type of the face_sizes is not correct : {}'.format( type(face_sizes) )
        if reset: self.reset(num_pts)
        else:     assert self.NUM_PTS == num_pts, 'The number of point is inconsistance : {} vs {}'.format(self.NUM_PTS, num_pts)

        print ('[GeneralDataset] load-data {:} datas begin'.format(len(datas)))

        for idx, data in enumerate(datas):
            assert isinstance(data, str), 'The type of data is not correct : {}'.format(data)
            assert osp.isfile(datas[idx]), '{} is not a file'.format(datas[idx])
            self.append(datas[idx], labels[idx], boxes[idx], face_sizes[idx])

        assert len(self.datas) == self.length, 'The length and the data is not right {} vs {}'.format(self.length, len(self.datas))
        assert len(self.labels) == self.length, 'The length and the labels is not right {} vs {}'.format(self.length, len(self.labels))
        assert len(self.face_sizes) == self.length, 'The length and the face_sizes is not right {} vs {}'.format(self.length, len(self.face_sizes))
        print ('Load data done for the general dataset, which has {} images.'.format(self.length))

    def load_list(self, file_lists, num_pts, reset):
        lists = load_file_lists(file_lists)
        print ('GeneralDataset : load-list : load {:} lines'.format(len(lists)))

        datas, labels, boxes, face_sizes = [], [], [], []

        for idx, data in enumerate(lists):
            alls = [x for x in data.split(' ') if x != '']
      
            assert len(alls) == 6 or len(alls) == 7, 'The {:04d}-th line in {:} is wrong : {:}'.format(idx, data)
            datas.append( alls[0] )
            if alls[1] == 'None':
                labels.append( None )
            else:
                labels.append( alls[1] )
            box = np.array( [ float(alls[2]), float(alls[3]), float(alls[4]), float(alls[5]) ] )
            boxes.append( box )
            if len(alls) == 6:
                face_sizes.append( None )
            else:
                face_sizes.append( float(alls[6]) )
            self.load_data(datas, labels, boxes, face_sizes, num_pts, reset)

    def __getitem__(self, index):
        assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
        image = pil_loader( self.datas[index] )
        target = self.labels[index].copy()
        return self._process_(image, target, index)

    def _process_(self, image, target, index):

        # transform the image and points
        if self.transform is not None:
            # image, lr_sm, lr_la, target = self.transform(image, target)
            # hr_im, lr_sm, image, target = self.transform(image, target)
            hr_im, lr_sm, lr_la, target = self.transform(image, target)

        # obtain the visiable indicator vector
        if target.is_none(): nopoints = True
        else               : nopoints = False

        # If for evaluation not load label, keeps the original data
        temp_save_wh = target.temp_save_wh
        ori_size = torch.IntTensor( [temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]] ) # H, W, Cropped_[x1,y1]
        
        if isinstance(hr_im, Image.Image):
            height, width = hr_im.size[1], hr_im.size[0]
        elif isinstance(hr_im, torch.FloatTensor):
            height, width = hr_im.size(1),  hr_im.size(2)
        else:
            raise Exception('Unknown type of image : {}'.format( type(hr_im) ))

        if target.is_none() == False:
            target.apply_bound(width, height)
            points = target.points.copy()
            points = torch.from_numpy(points.transpose((1,0))).type(torch.FloatTensor)
            Hpoint = target.points.copy()
        else:
            points = torch.from_numpy(np.zeros((self.NUM_PTS,3))).type(torch.FloatTensor)
            Hpoint = np.zeros((3, self.NUM_PTS))

        heatmaps, mask = generate_label_map(Hpoint, height//self.downsample, width//self.downsample, self.sigma, self.downsample, nopoints, self.heatmap_type) # H*W*C

        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
        mask     = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.ByteTensor)
      
        torch_index = torch.IntTensor([index])
        torch_nopoints = torch.ByteTensor( [ nopoints ] )


        return lr_sm, hr_im, lr_la, heatmaps, mask, points, torch_index, ori_size
'''