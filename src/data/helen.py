import os
from data import srdata

class HELEN(srdata.SRData):
    def __init__(self, args, name='HELEN', train=True, benchmark=False):
        # self.specificDataName = "HELEN_lr16_hr128_v0"        
        self.specificDataName = "HELEN_lr16_hr128_v1"
        # self.specificDataName = "HELEN_v0_TDAE"#"HELEN_v1_URDGN"#"HELEN_v1_FSRnet"
        # self.specificDataName = "HELEN_v1_SRRes"
        print("Dataset: {}".format(self.specificDataName))
        
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
        self.nParts = 194

        super(HELEN, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr, names_anno = super(HELEN, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        names_anno = names_anno[self.begin - 1:self.end]

        return names_hr, names_lr, names_anno

    def _set_filesystem(self, dir_data):
        super(HELEN, self)._set_filesystem(dir_data)
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'HELEN_train_HR')
            self.dir_lr = os.path.join(self.apath, 'HELEN_train_LR_bicubic')
            self.dir_anno = os.path.join(self.apath, 'HELEN_train_HR')
        else:
            self.dir_hr = os.path.join(self.apath, 'HELEN_test_HR')
            self.dir_lr = os.path.join(self.apath, 'HELEN_test_LR_bicubic')
            self.dir_anno = os.path.join(self.apath, 'HELEN_test_HR')
        
        if self.input_large: 
            if self.train:
                self.dir_lr = os.path.join(self.apath, 'HELEN_train_LR_large')
            else:
                self.dir_lr = os.path.join(self.apath, 'HELEN_test_LR_large')
