import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph import Sequential
import numpy as np


class ConvLSTMCell(fluid.dygraph.Layer):
    """
    基本cell结构
    """

    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = Conv2D(num_channels=self.input_dim + self.hidden_dim,
                           num_filters=4 * self.hidden_dim,
                           filter_size=self.kernel_size,
                           padding=self.padding,
                           param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = fluid.layers.concat([input_tensor, h_cur], axis=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = fluid.layers.split(combined_conv, 4, dim=1)
        i = fluid.layers.sigmoid(cc_i)
        f = fluid.layers.sigmoid(cc_f)
        o = fluid.layers.sigmoid(cc_o)
        g = fluid.layers.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * fluid.layers.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, b, h, w):
        # 初始化
        W = fluid.layers.create_parameter(shape=[b, self.hidden_dim, h, w], dtype='float32', 
                attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        B = fluid.layers.create_parameter(shape=[b, self.hidden_dim, h, w], dtype='float32', 
                attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        return (W, B)


class ConvLSTM(fluid.dygraph.Layer):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.cell_list = Sequential()
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            self.cell_list.add_sublayer(name='{}'.format(i), sublayer=ConvLSTMCell(in_channels=cur_input_dim,
                                                                                   hidden_channels=self.hidden_dim[i],
                                                                                   kernel_size=self.kernel_size[i],
                                                                                   ))

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list['{}'.format(layer_idx)](input_tensor=cur_layer_input[:, t, :, :, :],
                                                              cur_state=[h, c])
                output_inner.append(h)
            layer_output = fluid.layers.stack(output_inner, axis=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBLSTM(fluid.dygraph.Layer):
    # Constructor
    def __init__(self, in_channels, hidden_channels,
                 kernel_size, num_layers, batch_first=True):
        super(ConvBLSTM, self).__init__()
        self.forward_net = ConvLSTM(in_channels, hidden_channels, kernel_size,
                                    num_layers, batch_first=batch_first)

    def forward(self, xforward):
        """
        xforward, xreverse = B T C H W tensors.
        """

        y_out_fwd, _ = self.forward_net(xforward)

        y_out_fwd = y_out_fwd[-1]  # outputs of last CLSTM layer = B, T, C, H, W

        return y_out_fwd


from paddle.fluid.dygraph.base import to_variable

if __name__ == '__main__':
    # gradient check
    # convlstm = ConvLSTMCell(in_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3)
    with fluid.dygraph.guard():
        input = np.random.randn(5, 5, 1280, 1, 1).astype('float32')
        x = to_variable(input)

        model = ConvBLSTM(in_channels=1280, hidden_channels=512, kernel_size=(3, 3), num_layers=2)
        out = model(x)
        print(out.shape)
