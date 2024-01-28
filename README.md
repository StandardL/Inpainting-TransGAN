# Inpainting-TransGAN

Using ViTA Group's TransGAN to inpaint images.

Base code is [here](https://github.com/VITA-Group/TransGAN)



## Environment requirements

### Our environment

Ubuntu 16.04 with  8 Core CPU,  RTX 3090 24G*1 and 16G RAM

Python 3.10

### Guide

Use `pip` to install all the needed packages. You can just copy these command to install.

*First*, install all the package without pytorch.

```cmd
pip install imageio scipy==1.11.1 six numpy einops pillow python-dateutil==2.8.2 protobuf==3.19.0 tensorboard==1.12.2 tensorboardX==1.6 tqdm==4.29.1 opencv-python
```

*Then*, install pytorch 2.1.1. The command here is pytorch 2.1.1 with CUDA 11.8 version. If you are using different version of CUDA, you should search and download from pytorch's website.

```cmd
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

***Noted***

When you run the train script and it shows error below, please follow theses steps to install extra libraries.

1. ImportError: libGL.so.1: cannot open shared object file: No such file or directory

   Ubuntu Linux command: 

   ```cmd
   apt-get update
   apt install libgl1-mesa-glx
   ```



## Train or test

We try to use Trans-GAN to inpaint images on CelebA-HQ-256 and Place365. The original resolution of Place365 is 256\*256, and we have resized them to 128\*128, so if you want to use your own datasets to train Place365-inpainting, don't forget to resize them to 128 or write a new training script. 

For training CelebA-HQ-256, run `train_CelebA.py`, for training Place365, run `train_Place.py` .

Before training or testing, don't forget to modify the dataset path and mask path in `CelebAHQ_cfg.py`  and `Place365_128_cfg.py`. 

For testing, you should modify the checkpoint path in `CelebAHQ_cfg.py`  and `Place365_128_cfg.py`. 
