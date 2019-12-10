import random

import numpy as np
import skimage.color as sc
from scipy.ndimage.interpolation import zoom
import numbers, math
import copy
import torch
import cv2

def get_patch(*args, patch_size=128, scale=8, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]
    pts = args[2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy
    
    pts = pts - np.array([[tx],[ty],[0]])
    
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        args[1][ty:ty + tp, tx:tx + tp, :],
        pts
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(args[0]), _set_channel(args[1])]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        
        return tensor

    def _maps_np2Tensor(img):
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        # backward compatibility
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        tensor = img.float().div(255) 
        for i, (t, m, s) in enumerate(zip(tensor,mean,std)):
            t.sub_(m).div_(s)

        return tensor

    return [_np2Tensor(a) for a in args]
    # return [_np2Tensor(args[0]), _maps_np2Tensor(args[1]), _np2Tensor(args[2])]
    # return [_maps_np2Tensor(args[0]), _maps_np2Tensor(args[1]), _maps_np2Tensor(args[2])]

def augment(*args, hflip=True, rot=True):
    # lr, hr, pts
    hflip = hflip and random.random() < 0.5

    img_w = args[1].shape[1]    

    def _augment(img):
        if hflip: 
            img = img[:, ::-1, :]

        return img

    def pts_hflip(pts):
        num_of_pt = pts.shape[1]
        if num_of_pt == 68:
            flip_index = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, \
                            26, 25, 24, 23, 22, 21, 20, 19, 18, 17, \
                            27, 28, 29, 30, \
                            35, 34, 33, 32, 31, \
                            45, 44, 43, 42, 47, 46, \
                            39, 38, 37, 36, 41, 40, \
                            54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, \
                            64, 63, 62, 61, 60, 67, 66, 65]
        elif num_of_pt == 194:
            flip_index = [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, \
                            57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, \
                            70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, \
                            85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, \
                            99,  98,  97,  96,  95,  94,  93,  92,  91,  90,  89, 88,  87,  86, \
                            113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, \
                            134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, \
                            114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, \
                            174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, \
                            154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
        elif num_of_pt == 19:
            flip_index = [5, 4, 3, 2, 1, 0,\
                            11, 10, 9, 8, 7, 6,\
                            14, 13, 12,\
                            17, 16, 15, 18]
        temp = copy.deepcopy(pts)
        for i in range(0, num_of_pt):
            pts[0, flip_index[i]] = img_w - temp[0, i]
            pts[1, flip_index[i]] = temp[1, i]
            pts[2, flip_index[i]] = temp[2, i]
        return pts

    def _augment_pts(pts):
        if hflip: 
            pts = pts_hflip(pts)
        return pts


    return [_augment(args[0]), _augment(args[1]), _augment_pts(args[2]) ]

def augment_ori(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0#.5
    vflip = rot and random.random() < 0#.5
    rot90 = rot and random.random() < 0#.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

def load_txt_file(file_path):
    '''
    load data or string from text file.
    '''
    with open(file_path, 'r') as cfile:
        content = cfile.readlines()
    cfile.close()
    content = [x.strip() for x in content]
    num_lines = len(content)
    return content, num_lines

def anno_parser(anno_path, num_pts):  
    '''                        
    parse the annotation for 300W dataset, which has a fixed format for .pts file                                
    return:                    
    pts: 3 x num_pts (x, y, oculusion)                                
    '''                        
    data, num_lines = load_txt_file(anno_path)                          
    assert data[0].find('version: ') == 0, 'version is not correct'     
    assert data[1].find('n_points: ') == 0, 'number of points in second line is not correct'                     
    assert data[2] == '{' and data[-1] == '}', 'starting and end symbol is not correct'                          

    assert data[0] == 'version: 1' or data[0] == 'version: 1.0', 'The version is wrong : {}'.format(data[0])
    n_points = int(data[1][len('n_points: '):])                         

    assert num_lines == n_points + 4, 'number of lines is not correct'    # 4 lines for general information: version, n_points, start and end symbol      
    # assert num_pts == n_points, 'number of points is not correct'

    # read points coordinate   
    pts = np.zeros((3, n_points), dtype='float32')                      
    line_offset = 3    # first point starts at fourth line              
    point_set = set()
    for point_index in range(n_points):                                
        try:                     
            pts_list = data[point_index + line_offset].split(' ')       # x y format                                 
            if len(pts_list) > 2:    # handle edge case where additional whitespace exists after point coordinates   
                pts_list = remove_item_from_list(pts_list, '')              
            pts[0, point_index] = float(pts_list[0])                        
            pts[1, point_index] = float(pts_list[1])                        
            pts[2, point_index] = float(1)      # oculusion flag, 0: oculuded, 1: visible. We use 1 for all points since no visibility is provided by 300-W   
            point_set.add( point_index )
        except ValueError:       
            print('error in loading points in %s' % anno_path)              
    return pts, point_set

def anno_parser_v1(anno_path, NUM_PTS, one_base=False):
    '''
    parse the annotation for MUGSY-Full-Face dataset (AFLW), which has a fixed format for .pts file
    return: pts: 3 x num_pts (x, y, oculusion)
    '''
    data, n_points = load_txt_file(anno_path)
    assert n_points <= NUM_PTS, '{} has {} points'.format(anno_path, n_points)
    # read points coordinate
    pts = np.zeros((3, NUM_PTS), dtype='float32')
    point_set = set()
    for line in data:
        try:
            idx, point_x, point_y, oculusion = line.split(' ')
            idx, point_x, point_y, oculusion = int(idx), float(point_x), float(point_y), float(oculusion)
            if one_base==False: idx = idx+1
            assert idx >= 1 and idx <= NUM_PTS, 'Wrong idx of points : {:02d}-th in {:s}'.format(idx, anno_path)
            pts[0, idx-1] = point_x
            pts[1, idx-1] = point_y
            pts[2, idx-1] = float( oculusion )
            point_set.add(idx)
        except ValueError:
            raise Exception('error in loading points in {}'.format(anno_path))
    return pts, point_set

def faceSZ_from_pts(points):
    assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
    assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
    if points.shape[0] == 3:
        points = points[:2, points[-1,:].astype('bool') ]
    elif points.shape[0] == 2:
        points = points[:2, :]
    else:
        raise Exception('The shape of points is not right : {}'.format(points.shape))
    assert points.shape[1] >= 2, 'To get the box of points, there should be at least 2 vs {}'.format(points.shape)
    box = np.array([ points[0,:].min(), points[1,:].min(), points[0,:].max(), points[1,:].max() ])
    W = box[2] - box[0]
    H = box[3] - box[1]
    assert W > 0 and H > 0, 'The size of box should be greater than 0 vs {}'.format(box)

    faceSZ = np.sqrt(W * H)

    return faceSZ

def generate_label_map(pts, height, width, sigma, downsample, ctype):
    ## pts = 3 * N numpy array; points location is based on the image with size (height*downsample, width*downsample)
    #if isinstance(pts, numbers.Number):
    # this image does not provide the annotation, pts is a int number representing the number of points
    #return np.zeros((height,width,pts+1), dtype='float32'), np.ones((1,1,1+pts), dtype='float32')
    # nopoints == True means this image does not provide the annotation, pts is a int number representing the number of points

    assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[0] == 3, 'The shape of points : {}'.format(pts.shape)
    if isinstance(sigma, numbers.Number):
        sigma = np.zeros((pts.shape[1])) + sigma
    assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[1], 'The shape of sigma : {}'.format(sigma.shape)

    offset = downsample / 2.0 - 0.5
    num_points, threshold = pts.shape[1], 0.01

    visiable = pts[2, :].astype('bool')

    transformed_label = np.fromfunction( lambda y, x, pid : ((offset + x*downsample - pts[0,pid])**2 \
                                                        + (offset + y*downsample - pts[1,pid])**2) \
                                                          / -2.0 / sigma[pid] / sigma[pid],
                                                          (height, width, num_points), dtype=int)

    mask_heatmap      = np.ones((1, 1, num_points+1), dtype='float32')
    mask_heatmap[0, 0, :num_points] = visiable

    if ctype == 'laplacian':
        transformed_label = (1+transformed_label) * np.exp(transformed_label)
    elif ctype == 'gaussian':
        transformed_label = np.exp(transformed_label)
    else:
        raise TypeError('Does not know this type [{:}] for label generation'.format(ctype))
    transformed_label[ transformed_label < threshold ] = 0
    transformed_label[ transformed_label >         1 ] = 1
    transformed_label = transformed_label * mask_heatmap[:, :, :num_points]

    background_label  = 1 - np.amax(transformed_label, axis=2)
    background_label[ background_label < 0 ] = 0
    heatmap           = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')

    return heatmap*mask_heatmap, mask_heatmap

def resize_bi(im, scale, interp='INTER_CUBIC'):
    height, width = im.shape[:2]
    im_bi = cv2.resize(im, (width*scale, height*scale), interpolation=cv2.INTER_CUBIC)

    return im_bi

def apply_bound(points, width, height):
    oks = np.vstack((points[0, :] >= 0, 
                    points[1, :] >=0, 
                    points[0, :] <= width, 
                    points[1, :] <= height, 
                    points[2, :].astype('bool')))
    oks = oks.transpose((1,0))
    points[2, :] = np.sum(oks, axis=1) == 5
    return points

