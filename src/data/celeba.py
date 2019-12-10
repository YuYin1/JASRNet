import os
from data import srdata

class CelebA(srdata.SRData):
    def __init__(self, args, name='CelebA', train=True, benchmark=False):
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
        self.nParts = 5
        # self.heatmap_sz = 32
        super(CelebA, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr, names_anno = super(CelebA, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        names_anno = names_anno[self.begin - 1:self.end]

        return names_hr, names_lr, names_anno

    def _set_filesystem(self, dir_data):
        super(CelebA, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'CelebA_train_HR')
        self.dir_lr = os.path.join(self.apath, 'CelebA_train_LR_bicubic')
        self.dir_anno = os.path.join(self.apath, 'CelebA_train_HR')
        
        if self.input_large: self.dir_lr += 'L'

