from torch import nn
import numpy as np
import torch


class ModelArrhythmia(nn.Module):
    def __init__(self, input_shape, output_shape, n_blocks, init_channel, kernel_size, dilation):
        super(ModelArrhythmia, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.channel = init_channel
        self.n_layers = n_blocks
        self.l_in, self.l_out = 0, 0
        self.stride_inc = 2  # default = 2
        self.channel_inc = 4  # default = 4
        self.fcchannel=[1024,256]

        self.base_block = nn.ModuleList()
        self.block = nn.ModuleList()
        self.skip = nn.ModuleList()

        # BaseBlock
        self._base_block()
        # MainBlock
        self._main_block()

    def _base_block(self):
        # [0] conv
        stride = 1
        self.l_in = self.input_shape[1]
        self.l_out = int(self.l_in / stride)
        padding = self._padding(self.l_in, self.l_out, self.kernel_size, stride, self.dilation)
        self.base_block.append(self._conv_block(in_channels=self.input_shape[0], out_channels=self.channel,
                                                act='relu',
                                                bn=True,
                                                dropout=False,
                                                kernel_size=self.kernel_size,
                                                dilation=self.dilation,
                                                stride=stride,
                                                padding=padding))
        # [1] skip connection (max pool)
        padding = int(np.ceil((2 * ((self.l_out / 2) - 1) - self.l_out + 2) / 2))
        self.base_block.append(self._max_pool1d(kernel_size=2, padding=padding))

        # [2] conv
        stride = 2
        self.l_in = self.l_out
        self.l_out = int(self.l_in / stride)
        padding = self._padding(self.l_in, self.l_out, self.kernel_size, stride, self.dilation)
        self.base_block.append(self._conv_block(in_channels=self.channel, out_channels=self.channel,
                                                act='relu',
                                                bn=True,
                                                dropout=True,
                                                kernel_size=self.kernel_size,
                                                dilation=self.dilation,
                                                stride=stride,
                                                padding=padding))
        # [3]
        stride = 1
        self.l_in = self.l_out
        self.l_out = int(self.l_in / stride)
        padding = self._padding(self.l_in, self.l_out, self.kernel_size, stride, self.dilation)
        self.base_block.append(self._conv_block(in_channels=self.channel, out_channels=self.channel,
                                                act='relu',
                                                bn=False,
                                                dropout=False,
                                                kernel_size=self.kernel_size,
                                                dilation=self.dilation,
                                                stride=stride,
                                                padding=padding))

    def _main_block(self):
        for i in range(self.n_layers):
            # [0] Main
            self.block.append(nn.Sequential(
                nn.BatchNorm1d(self.channel),
                nn.ReLU()
            ))

            in_channels = self.channel
            if i % self.channel_inc == 0:
                self.channel *= 2
            if i % self.stride_inc == 0:
                stride = 2
            else:
                stride = 1
            self.ori_len = self.l_out
            self.l_in = self.l_out
            self.l_out = int(self.l_in / stride)
            padding = self._padding(self.l_in, self.l_out, self.kernel_size, stride, self.dilation)
            # [1] Main (conv1d)
            self.block.append(self._conv_block(in_channels=in_channels,
                                               out_channels=self.channel,
                                               act='relu',
                                               bn=True,
                                               dropout=True,
                                               kernel_size=self.kernel_size,
                                               dilation=self.dilation,
                                               stride=stride,
                                               padding=padding))

            stride = 1
            padding = self._padding(self.l_out, self.l_out, self.kernel_size, stride, self.dilation)
            # [2] Main (conv1d)
            self.block.append(self._conv_block(in_channels=self.channel,
                                               out_channels=self.channel,
                                               act='relu',
                                               bn=False,
                                               dropout=False,
                                               kernel_size=self.kernel_size,
                                               dilation=self.dilation,
                                               stride=stride,
                                               padding=padding))
            # [3] Skip connection (max pooling)
            stride = 2 if i % self.stride_inc == 0 else 1
            l_out = int(self.ori_len / stride)
            padding = self._padding(self.ori_len, l_out, stride, stride, 1)
            self.block.append(self._max_pool1d(kernel_size=stride, padding=padding))

        classifier = nn.Sequential(
            nn.BatchNorm1d(self.channel),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(self.l_out*self.channel,self.fcchannel[0]),
                    nn.Linear(self.fcchannel[0],self.fcchannel[1]),
                    nn.Linear(self.fcchannel[1], self.output_shape),
                    nn.Dropout(0.2))
        self.block.append(classifier)

    def forward(self, xb):
        # Base block
        xb = self.base_block[0](xb)
        skip = self.base_block[1](xb)  # Skip connection
        xb = self.base_block[2](xb)
        xb = self.base_block[3](xb)
        xb = torch.add(xb, skip)  # Adding output with skip connection

        num_comp = 4
        for i in range(self.n_layers):
            skip = xb
            xb = self.block[i * num_comp + 0](xb)
            skip = self.block[i * num_comp + 3](skip)

            xb = self.block[i * num_comp + 1](xb)
            xb = self.block[i * num_comp + 2](xb)
            if i % self.channel_inc == 0:
                # Concatenating zero-padding in skip-connection to match the dimension of channels
                if torch.cuda.is_available():
                    zeros = torch.zeros(skip.shape, device='cuda', dtype=torch.float, requires_grad=True)
                else:
                    zeros = torch.zeros(skip.shape, dtype=torch.float, requires_grad=True)
                skip = torch.cat((skip, zeros), dim=-2)  # Along the channels
            xb = torch.add(xb, skip)
        xb = self.block[-1](xb)
        return xb

    @staticmethod
    def _conv_block(in_channels, out_channels, act, bn, dropout, *args, **kwargs):
        """
        1-D Convolution output l_out can be defined as
        l_out = (l_in + 2P - (K - 1) - 1) / S + 1,
        where P, K, S denote 'padding', 'kernel_size', and 'stride', respectively.
        """
        modules = nn.ModuleList([nn.Conv1d(in_channels, out_channels, *args, **kwargs)])
        if bn:
            modules.append(nn.BatchNorm1d(out_channels))
        if act == 'relu':
            modules.append(nn.ReLU())
        if dropout:
            modules.append(nn.Dropout(p=.2))

        net = nn.Sequential(*modules)
        return net

    @staticmethod
    def _padding(l_in, l_out, kernel_size, stride, dilation):
        return int(np.ceil((stride * (l_out - 1) - l_in + dilation * (kernel_size - 1) + 1) / 2))

    @staticmethod
    def _max_pool1d(*args, **kwargs):
        modules = nn.ModuleList([nn.MaxPool1d(*args, **kwargs)])
        return nn.Sequential(*modules)


if __name__ == '__main__':
    model = ModelArrhythmia(input_shape=[12, 5000],
                        output_shape=9,
                        n_blocks=15,
                        init_channel=32,
                        kernel_size=15,
                        dilation=1) # CLASS, CHANNEL, TIMEWINDOW
    # pred = model(torch.zeros(50, 1, 20, 250))
    # print(model)
    from pytorch_model_summary import summary

    print(summary(model, torch.zeros((1, 12, 5000)), show_input=False))
            # (1, 1, channel, timewindow)
