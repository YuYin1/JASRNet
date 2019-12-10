# Joint Super-resolution and Alignment of Tiny Faces (JASRNet)
This repository contains the code for our paper [Joint Super-resolution and Alignment of Tiny Faces](https://arxiv.org/abs/1911.08566) (**AAAI2020**).

## Train
For training, run 
`python main.py --save_results --save_gt --save_models`.

## Test
For testing, run
`python main.py --save test_folder --test_only --save_results --save_gt --pre_train ../experiment/model.pth`.


## Citation
Please cite this paper in your publications if it helps your research:

>@article{yin2019joint,  
  title={Joint Super-Resolution and Alignment of Tiny Faces},  
  author={Yin, Yu and Robinson, Joseph P and Zhang, Yulun and Fu, Yun},  
  journal={arXiv preprint arXiv:1911.08566},  
  year={2019}  
}
