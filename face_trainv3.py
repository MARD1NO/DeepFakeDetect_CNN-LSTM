import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.base import to_variable
import math
from FaceDataLoader import *
from CNNRNNModel import CNN_RNN_Model2
from CNNRNNModel2 import CNN_RNN_Model3

def get_lr(base_lr = 0.01, lr_decay = 0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    return learning_rate
    
if __name__ == "__main__":
    train_video_path = "/home/aistudio/work/train_sample_videos"
    meta_data_path = "/home/aistudio/work/train_sample_videos/metadata.json"
    test_video_path = "/home/aistudio/work/test_videos"
    # threshold = 0.3 # 用于对过多的fake video采样
    face_data_dir = '/home/aistudio/work/face_image'
    resolution = 224
    batchsize = 8
    frame_length = 10
    with fluid.dygraph.guard():
        model = CNN_RNN_Model3()
        opt = fluid.optimizer.AdamOptimizer(learning_rate=0.0001,  regularization=fluid.regularizer.L2Decay(0.00001),  parameter_list=model.parameters())  #创建优化器
        
        train_loader = BatchedDataLoaderv2(face_data_dir, resolution, batchsize, frame_length)
        video_file_count = len(os.listdir(face_data_dir))
        max_per_epoch = video_file_count // batchsize
        print("max per epoch is", max_per_epoch)
        MAX_EPOCH = 200
        for epoch in range(MAX_EPOCH):
            smallepoch = 1
            total_train_video_sample = 0
            accurate_count = 0
            total_loss = 0
            total_acc = 0
            for i, data in enumerate(train_loader()):
                if smallepoch > max_per_epoch - 1:
                    break
                
                frame_data, frame_label= data
                
                # 转化为合适的数据类型
                frame_data = np.array(frame_data).astype('float32')
                frame_label = np.array(frame_label).astype('int64')
                
                img = fluid.dygraph.to_variable(frame_data, name="img")
                label = fluid.dygraph.to_variable(frame_label, name="label")
                label.stop_gradient = True 
                output, acc = model(img, label)
                # 计算交叉熵loss
                loss = fluid.layers.cross_entropy(output, label)
                mean_loss = fluid.layers.reduce_mean(loss)
                mean_loss.backward()
                total_loss += mean_loss
                opt.minimize(mean_loss)
                model.clear_gradients()
                total_acc += acc.numpy()[0]
                
                smallepoch += 1
                total_train_video_sample += batchsize
            
            print("EPOCH ", epoch)
            print("total loss is ", sum(total_loss.numpy()))
            print("Accuracy is ", total_acc/smallepoch)
    
            if epoch % 50 == 0 or epoch == MAX_EPOCH-1:
                fluid.save_dygraph(model.state_dict(), 'DpDcModel_LSTM2_epoch{}'.format(epoch))
