import os
from data import srdata

class W300(srdata.SRData):
    def __init__(self, args, name='W300', train=True, benchmark=False):
        self.specificDataName = "300W_lr16_hr128_v1"
        # self.specificDataName = "300W_lr64_hr256_v1"
        # self.specificDataName = "300W_lr128_hr256_v1"

        # self.specificDataName = "300W_lr28_hr224"
        # self.specificDataName = "300W_lr224_hr448"

        # self.specificDataName = "300W_lr16_hr128_v1_common"
        # self.specificDataName = "300W_lr16_hr128_v1_chan"
        # self.specificDataName = "300W_lr64_hr256_v1_common"
        # self.specificDataName = "300W_lr64_hr256_v1_chan"
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
        args.nParts = 68
        self.nParts = args.nParts
        super(W300, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr, names_anno = super(W300, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        names_anno = names_anno[self.begin - 1:self.end]

        return names_hr, names_lr, names_anno

    def _set_filesystem(self, dir_data):
        super(W300, self)._set_filesystem(dir_data)
        if self.train:
            self.dir_hr = os.path.join(self.apath, '300W_train_HR')
            self.dir_lr = os.path.join(self.apath, '300W_train_LR_bicubic')
            self.dir_anno = os.path.join(self.apath, '300W_train_HR')
        else:
            self.dir_hr = os.path.join(self.apath, '300W_test_HR')
            self.dir_lr = os.path.join(self.apath, '300W_test_LR_bicubic')
            self.dir_anno = os.path.join(self.apath, '300W_test_HR')
        
        if self.input_large: self.dir_lr += 'L'

