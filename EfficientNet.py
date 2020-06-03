import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.initializer import Xavier, Constant
import math
from model_utils import *
import collections
import numpy as np


class Swish(fluid.dygraph.Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return fluid.layers.hard_swish(x)


class MBConvBlock(fluid.dygraph.Layer):
    def __init__(self, block_args, global_params, image_size=None):
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._bn_mom = global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels

        if self._block_args.expand_ratio != 1:
            self._expand_conv = fluid.dygraph.Conv2D(
                num_channels=inp,
                num_filters=oup,
                filter_size=1,
                stride=1,
                padding=0,
                # param_attr=ParamAttr(name='expand_conv_w', initializer=Xavier(uniform=False)),
                bias_attr=False
            )

            self._bn0 = fluid.dygraph.BatchNorm(
                num_channels=oup,
                # param_attr=ParamAttr(name='bn0_w', initializer=Xavier(uniform=False)),
                momentum=self._bn_mom,
                epsilon=self._bn_eps,
            )

        k = self._block_args.kernel_size
        s = self._block_args.stride  # 由于返回的是列表，这里取0
        self._depthwise_conv = fluid.dygraph.Conv2D(
            num_channels=oup,
            num_filters=oup,
            filter_size=k,
            stride=s[0],
            padding=int((k - 1) // 2),
            groups=oup,
            bias_attr=False
        )
        self._bn1 = fluid.dygraph.BatchNorm(
            num_channels=oup,
            # param_attr=ParamAttr(name='bn1_w', initializer=Xavier(uniform=False)),
            momentum=self._bn_mom,
            epsilon=self._bn_eps,
        )

        if self.has_se:
            num_reduced_filters = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = fluid.dygraph.Conv2D(
                num_channels=oup,
                num_filters=num_reduced_filters,
                filter_size=1,
                stride=1,
                padding=0)

            self._se_expand = fluid.dygraph.Conv2D(
                num_channels=num_reduced_filters,
                num_filters=oup,
                filter_size=1,
                stride=1,
                padding=0)

        final_oup = self._block_args.output_filters
        self._project_conv = fluid.dygraph.Conv2D(
            num_channels=oup,
            num_filters=final_oup,
            filter_size=1,
            stride=1,
            padding=0,
            bias_attr=False
        )

        self._bn2 = fluid.dygraph.BatchNorm(
            num_channels=final_oup,
            momentum=self._bn_mom,
            epsilon=self._bn_eps,
        )
        self._swish = Swish()

        self._avg_pool = fluid.dygraph.Pool2D(pool_type='avg', global_pooling=True)

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.has_se:
            x_squeezed = self._avg_pool(x)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = fluid.layers.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, survival_prob=1 - drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(fluid.dygraph.Layer):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._blocks = []
        # Batch norm parameters
        bn_mom = self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = fluid.dygraph.Conv2D(
            num_channels=in_channels,
            num_filters=out_channels,
            filter_size=3,
            stride=2,
            padding=(3 - 1) // 2)
        self._bn0 = fluid.dygraph.BatchNorm(num_channels=out_channels,
                                            momentum=bn_mom,
                                            epsilon=bn_eps, )

        for i, block_args in enumerate(self._blocks_args):
            assert block_args.num_repeat > 0
            j = 0
            # Update block input and output filters based on depth multiplier.

            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            sublayer = MBConvBlock(block_args, self._global_params)
            self._blocks.append(sublayer)
            self.add_sublayer('MBConvBlock_' + str(i) + str(j), sublayer)

            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                block_args = block_args._replace(input_filters=block_args.output_filters,
                                                 stride=[1])
            for _ in range(block_args.num_repeat - 1):
                j += 1
                sublayer = MBConvBlock(block_args, self._global_params)
                self._blocks.append(sublayer)
                self.add_sublayer('MBConvBlock_' + str(i) + str(j), sublayer)

        # 这里是不是写错了
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = fluid.dygraph.Conv2D(
            num_channels=in_channels,
            num_filters=out_channels,
            filter_size=1,
            stride=1,
            padding=0,
        )

        self._bn1 = fluid.dygraph.BatchNorm(
            num_channels=out_channels,
            momentum=bn_mom,
            epsilon=bn_eps,
        )

        self._avg_pooling = fluid.dygraph.Pool2D(
            pool_type='avg',
            global_pooling=True,
        )

        self._fc = Linear(out_channels, self._global_params.num_classes)
        self._swish = Swish()

    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs, label=None):

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = fluid.layers.flatten(x)
        x = fluid.layers.dropout(x, self._global_params.dropout_rate)
        x = self._fc(x)
        return x

from paddle.fluid.dygraph.base import to_variable

if __name__ == "__main__":
    model_name = "efficientnet-b0"
    override_params = {"num_classes": 12}
    blocks_args, global_params = get_model_params(model_name, override_params=override_params)

    print(blocks_args)

    with fluid.dygraph.guard():
        x = np.random.randn(1, 3, 224, 224).astype('float32')
        x = to_variable(x)
        net = EfficientNet(blocks_args, global_params)

        out = net(x)
        print(out.shape)
