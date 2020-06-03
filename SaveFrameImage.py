"""
数据预处理
将meatadata里的数据转化成一个列表
列表里面每个元素都是一个字典，形如{'video_file':'xxxx', 'label':0/1}
label = 0 -> FAKE
label = 1 -> REAL
"""
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

def metadataReader(label_path) -> List:
    """
    数据预处理
    将meatadata里的数据转化成一个列表
    列表里面每个元素都是一个字典，形如{'video_file':'xxxx', 'label':0/1}
    label = 0 -> FAKE
    label = 1 -> REAL
    :param label_path: metadata 文件路径
    :return: record
    """
    record = []

    with open(label_path) as f:
        file = f.read()
        jsonfile = json.loads(file)
        for key, val in zip(jsonfile.keys(), jsonfile.values()):
            rec = {'video_file': key,
                   'label': [1] if val['label'] == "REAL" else [0]}
            record.append(rec)

    return record


def ImageResize(img_array: 'np.ndarray', resize: int) -> np.ndarray:
    """
    输入ndarray，转化为RGB 并进行resize
    :param img_array: 输入的图片，格式为ndarray
    :param resize: 缩放大小
    :return:
    """
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # 转为rgb
    return cv2.resize(img_array, (resize, resize))


def random_distort(img):
    """
    随机改变亮度，对比度，颜色
    :param img:
    :return:
    """

    # 随机改变亮度
    def random_brightness(img, lower=0.75, upper=1.25):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    # 随机改变对比度
    def random_contrast(img, lower=0.75, upper=1.25):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    # 随机改变颜色
    def random_color(img, lower=0.75, upper=1.25):
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

def CaptureVideoImage(
        videoFile: str,
        savedir: str, 
        label:int,
        totalFrame=15,
        resolution=640) -> np.ndarray:
    """
    读取videoFile，生成一个[totalFrame, resolution, resolution, channel]形式的数组
    :param videoFile: 视频文件
    :param totalFrame: 截取总共帧数，作为一个batch
    :param resolution: 分辨率，默认用efficientnetb0，参数为224
    :return: [totalFrame, resolution, resolution, channel] ndarray

    """
    video = np.zeros((totalFrame, resolution, resolution, 3), dtype='int')  # 存储数组
    cnt = 0  # 数组计数器
    vidcap = cv2.VideoCapture(videoFile)  # 视频流截图
    frame_all = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    frame_start = random.randint(0, frame_all // 2)  # 起始帧

    frame_interval = 2

    if vidcap.isOpened():
        for i in range(frame_start, frame_start + totalFrame * frame_interval, frame_interval):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)  # set方法获取指定帧
            success, img = vidcap.read()
            if success:
                img = ImageResize(img, resolution)
                img_save_dir = savedir+"{}_{}.jpg".format(label, i)
                print(img_save_dir)
                cv2.imwrite(img_save_dir, img)
                
               

def Data_Loader(data_dir, video_dir, savedir):
    """
    数据装载器
    :param data_dir: metadata.json 路径
    :param video_dir: 视频文件路径
    :return: reader闭包函数
    yield 标签，帧数据


    """
    records = metadataReader(data_dir)
    video_dir = video_dir
    threshold = 0.25 # 对fake video进一步采样，以保证均衡
    
    epochs = 2

    for epoch in range(epochs):
        for record in records:
            video_file_fullname = os.path.join(video_dir + '/', record['video_file'])
            video_file_label = record['label']
            video_file_label = video_file_label[0]

            if video_file_label == 0:
                randnum = random.randint(0, 100) / 100
                if randnum > threshold:
                    continue

            video_dir_name = record['video_file'].split('.')[0] # 取abcd.mp4中去掉.mp4后缀，作为文件夹名字
            video_dir_name += '_' + str(epoch) +'_' + str(video_file_label)
            os.mkdir(savedir+'/'+ video_dir_name)
            video_folder_name = savedir+'/'+video_dir_name + '/'
            print("now video folder name is ", video_folder_name)
            # now video folder name is  /home/aistudio/work/frameimage/aapnvogymq_0_0/ 
            # 第一个数字是epoch，第二个数字是标签
            
            video_frame_data = CaptureVideoImage(video_file_fullname, video_folder_name, video_file_label)
        

if __name__ == "__main__":
    train_video_path = "/home/aistudio/work/train_sample_videos"
    save_dir = "/home/aistudio/work/validate_frame_image"
    meta_data_path = "/home/aistudio/work/train_sample_videos/metadata.json"
    Data_Loader(meta_data_path, train_video_path, save_dir)
