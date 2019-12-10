import argparse

parser = argparse.ArgumentParser(description='JASRNet')

parser.add_argument('--debug', action='store_true', help='Enables debug mode')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../Dataset', help='dataset directory')
parser.add_argument('--data_train', type=str, default='300W', help='train dataset name')
parser.add_argument('--data_test', type=str, default='300W', help='test dataset name')
parser.add_argument('--nParts', type=int, default=68, help='num of landmarks')
parser.add_argument('--data_range', type=str, default='1-47220/1-689', help='train/test data range')
parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')
parser.add_argument('--scale', type=str, default='8', help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=128, help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true', help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='JASR', help='model name')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.', help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=32, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=128, help='number of feature maps')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--test_every', type=int, default=100, help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=5.0e-5, help='learning rate')
parser.add_argument('--decay', type=str, default='20-30', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*CPM+1*L1', help='loss function configuration') 
parser.add_argument('--skip_threshold', type=float, default='1e8', help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test', help='file name to save')
parser.add_argument('--load', type=str, default='', help='file name to load')
parser.add_argument('--resume', type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=1000, help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', help='save output results')
parser.add_argument('--save_gt', action='store_true', help='save low-resolution and high-resolution images together')

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

