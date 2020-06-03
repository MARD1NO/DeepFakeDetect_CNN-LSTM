import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph.base import to_variable
import math
from EfficientNet import *
from model_utils import *
from convlstm import ConvBLSTM
from paddle.fluid.dygraph import Linear


class CNNEnoder(fluid.dygraph.Layer):
    def __init__(self, fc_hidden1=256, fc_hidden2=128, drop_p=0.5, CNN_embed_dim=300):
        super(CNNEnoder, self).__init__()
        model_name = "efficientnet-b0"
        override_params = {"num_classes": 1024}
        blocks_args, global_params = get_model_params(model_name, override_params=override_params)
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.BackBone = EfficientNet(blocks_args, global_params)

        self.fc1 = Linear(override_params["num_classes"], fc_hidden1, 
                        param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.01),
                regularizer=L2Decay(0.)))
        self.bn1 = BatchNorm(num_channels=fc_hidden1, momentum=0.9, 
                        param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0., 0.01),
                                regularizer=L2Decay(0.)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0),
                                regularizer=L2Decay(0.)))
        self.fc2 = Linear(fc_hidden1, fc_hidden2, 
                        param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0., 0.02),
                                regularizer=L2Decay(0.)))
        self.bn2 = BatchNorm(num_channels=fc_hidden2, momentum=0.9, 
                        param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0., 0.01),
                                regularizer=L2Decay(0.)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0),
                                regularizer=L2Decay(0.)))
        self.fc3 = Linear(fc_hidden2, CNN_embed_dim, 
                        param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0., 0.01),
                                regularizer=L2Decay(0.)))

    def forward(self, x_3d):
        cnn_embed_seq = []
        for time in range(x_3d.shape[1]):
            x = self.BackBone(x_3d[:, time, :, :, :])
            # print(x.shape)
            x = fluid.layers.flatten(x, axis=1)
            # print("x flatten shape is ", x.shape)
            x = self.bn1(self.fc1(x))
            x = fluid.layers.relu(x)
            x = self.bn2(self.fc2(x))
            x = fluid.layers.relu(x)
            x = fluid.layers.dropout(x, self.drop_p)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        cnn_embed_seq = fluid.layers.stack(cnn_embed_seq, axis=0)
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        # print(cnn_embed_seq.shape)
        cnn_embed_seq = fluid.layers.transpose(cnn_embed_seq, perm=[1, 0, 2])
        cnn_embed_seq = fluid.layers.unsqueeze(input=cnn_embed_seq, axes=[3, 4])
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # print(cnn_embed_seq.shape)
        # print("cnn shape",  cnn_embed_seq.shape)
        return cnn_embed_seq


class CNN_RNN_Model2(fluid.dygraph.Layer):
    def __init__(self):
        super(CNN_RNN_Model2, self).__init__()
        self.EfficientNet = CNNEnoder(CNN_embed_dim=512)
        self.RNN =ConvBLSTM(in_channels=512, hidden_channels=256, kernel_size=(3, 3), num_layers=1)
        self.fc1 = Linear(1280, 512, param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        self.fc2 = Linear(512, 2, act='softmax')

    def forward(self, x, label=None):
        x = self.EfficientNet(x)
        x = self.RNN(x)
        x = fluid.layers.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = fluid.layers.relu(x)
        y = self.fc2(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y
            
if __name__ == "__main__":        
    with fluid.dygraph.guard():
        # batch_size, 10(frames), 3, 240, 240
        x = np.random.randn(2, 10, 3, 224, 224).astype('float32')
        x = to_variable(x)
        model = CNN_RNN_Model2()

        out = model(x)
        print(out.shape)
        # print(out)
