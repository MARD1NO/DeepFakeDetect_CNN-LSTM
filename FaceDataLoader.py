import cv2
import matplotlib.pyplot as plt
import paddle
import numpy as np
from PIL import Image, ImageEnhance
import random
import os
import json
from typing import *
import functools
from matplotlib.image import imread
from random import random
from scipy.ndimage.filters import gaussian_filter

def gaussian_blur(img, sigma):
    # 高斯平滑
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def cv2_jpg(img, compress_val):
    # jpeg压缩
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def data_augment_deepfake(img):
    # 根据论文设定的相关数据增广方法
    img = np.array(img)
    blur_prob = 0.9
    jpg_prob = 0.9
    
    if random() < blur_prob:
        sig = np.random.uniform(0, 3)
        gaussian_blur(img, sig)

    if random() < jpg_prob:
        img = cv2_jpg(img, 75)
    return img

def random_distort(img):
    """
    随机改变亮度，对比度，颜色
    :param img:
    :return:
    """

    # 随机改变亮度
    def random_brightness(img, lower=0.8, upper=1.2):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    # 随机改变对比度
    def random_contrast(img, lower=0.8, upper=1.2):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    # 随机改变颜色
    def random_color(img, lower=0.8, upper=1.2):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img

def random_flip(img, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :]
        
    return img


def Normalize_Img(img):
    """
    对图片数据做归一化
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))
    return img
 
def Image_augumentation(img, resolution):
    # img = random_distort(img)
    # img = random_flip(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转成RGB先
    img = cv2.resize(img, (resolution, resolution))
    img = data_augment_deepfake(img)
    
    img = img.astype('float32')
    img = (img / 255.0) # 只除255做归一化
    img = img.transpose((2, 0, 1))
    return img
 

def Data_Loaderv2(data_dir, resolution, frame_length):
    """
    数据装载器
    :param data_dir: metadata.json 路径
    :param video_dir: 视频文件路径
    :param frame_length:每个文件取多少帧
    :return: reader闭包函数
    yield 标签，帧数据
    """
    # datadir is faceimage/
    framfileDirs = os.listdir(data_dir)
    # threshold = 0.25 # 对fake video进一步采样，以保证均衡
    def reader():
        for framefileDir in framfileDirs:
            # frameDir is dhoqofwoxa
            if framefileDir == ".ipynb_checkpoints":
                continue
            perVideoFrameDir = os.path.join(data_dir, framefileDir) # /faceimage/dhoqofwoxa
            # print(perVideoFrameDir)
            perImageDirList = os.listdir(perVideoFrameDir)
            # 随机打乱文件顺序
            np.random.shuffle(perImageDirList)
            perImageDirList.sort(key=lambda x: int(x.split('_')[1][:-4]))
            # print(perImageDirList)
            videoArray = np.zeros((frame_length, 3, resolution, resolution), dtype='float32') # [10, 3, 224, 224]
            
            cnt = 0
            
            for perImageDir in perImageDirList:
                if cnt == frame_length:
                    break

                fullImagedir = os.path.join(perVideoFrameDir, perImageDir)
                # print("now it is ", fullImagedir)
                img = cv2.imread(fullImagedir)
                # print(img.shape)
                img = Image_augumentation(img, resolution)
                # print("img shape is ", img.shape)
                videoArray[cnt] = img    
                label = perImageDir.split('_')[0]
                cnt += 1
            
            if label == '0':
                labelArray = np.array([0])
                # labelArray = np.array([1, 0])
            else:
                labelArray = np.array([1])
                # labelArray = np.array([0, 1])
            
            yield (videoArray, labelArray)

    return reader

    
def BatchedDataLoaderv2(data_dir, resolution, batchsize, frame_length):
    train_loader = Data_Loaderv2(data_dir, resolution, frame_length)
    # video_file_count = len(os.listdir(data_dir))
    # max_epoch = video_file_count // batchsize
    def reader():
        videolist = []
        labellist = []
        for data in train_loader():
            videolist.append(data[0])
            labellist.append(data[1])
            if len(labellist) == batchsize:
                # print("Success Load Image!")
                videoarray = np.array(videolist)
                labelarray = np.array(labellist)
                yield (videoarray, labelarray)
                videolist = []
                labellist = []
    return reader

if __name__ == "__main__":
    data_dir = '/home/aistudio/work/faceimage_small'
    reader = BatchedDataLoaderv2(data_dir, 224, 5, 10)
    # reader = Data_Loaderv2(data_dir, 224, 10)
    videoarray, labelarray = next(reader())
    print("video array shape is ", videoarray.shape)
    print("label Array shape is", labelarray.shape)
    print(labelarray[0])
    img = videoarray[0, 1, :, :, :]
    img = img.transpose(2, 1, 0)
    print(img)
    print(img.shape)
    plt.imshow(img)
    plt.show()
