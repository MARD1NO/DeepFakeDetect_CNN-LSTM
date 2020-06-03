import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.base import to_variable
import math
from FaceDataLoader import *
from CNNRNNModel import CNN_RNN_Model2
from CNNRNNModel2 import CNN_RNN_Model3
import argparse
import re

def get_lr(base_lr = 0.01, lr_decay = 0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    return learning_rate
    
def parse_args():
    parser = argparse.ArgumentParser("Evaluation Parameters")
    parser.add_argument(
        '--weight_file',
        type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    WEIGHT_FILE = args.weight_file


    train_video_path = "/home/aistudio/work/train_sample_videos"
    meta_data_path = "/home/aistudio/work/train_sample_videos/metadata.json"
    test_video_path = "/home/aistudio/work/test_videos"
    face_data_dir = '/home/aistudio/work/validate_face_image'
    resolution = 224
    frame_length = 10
    batchsize = 1
    total_acc = 0
    smallepoch = 0
    with fluid.dygraph.guard():
        model = CNN_RNN_Model3()
        
        params_file_path = WEIGHT_FILE
        print("NOW LOADING {} Params".format(params_file_path))
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.set_dict(model_state_dict)
        model.eval()

        train_loader = BatchedDataLoaderv2(face_data_dir, resolution, batchsize, frame_length)
        
        accs = []
        for i, data in enumerate(train_loader()):
            smallepoch += 1            
            frame_data, frame_label= data
            
            frame_data = np.array(frame_data).astype('float32')
            frame_label = np.array(frame_label).astype('int64')
            
            img = fluid.dygraph.to_variable(frame_data, name="img")
            label = fluid.dygraph.to_variable(frame_label, name="label")
            label.stop_gradient = True
            output, acc = model(img, label)
            print("output is", output.numpy())
            accs.append(acc.numpy()[0])
            
        print("ACC is:", sum(accs)/len(accs))
