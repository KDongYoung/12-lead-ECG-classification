from numpy.core.shape_base import block
from torch import nn
import torch
from torch.nn.modules.dropout import Dropout

class ModelArrhythmia(nn.Module):
    def __init__(self, input_shape, output_shape, block_num, kernel_size, stride, padding):
        super(ModelArrhythmia, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.channel = [32,64,128,256,512]
        self.fcchannel = [1024,256]
        self.stride = stride  # default = 2
        self.padding = padding  # default = 4
        self.block_num=block_num

        self.block = nn.ModuleList()
        self.block2 = nn.ModuleList()
        
        # MainBlock
        self._main_block()

        self.block.eval()
        out = self.block(torch.zeros(1, self.input_shape[0], self.input_shape[1]))
        self.n_outputs = out.size()[0] * out.size()[1] * out.size()[2]
        # self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.output_shape), nn.Dropout(p=0.2))
        self.lstm_attention_block()


    def _main_block(self):
        # [0] Main (conv1d)
        self.block.append(self._conv_block(in_channels=self.input_shape[0],
                                            out_channels=self.channel[1],
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            stride=self.stride,
                                            n_layers=2,
                                            repeat=1))
        
        # [1] Main (conv1d)
        self.block.append(self._conv_block(in_channels=self.channel[1],
                                            out_channels=self.channel[2],
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            stride=self.stride,
                                            n_layers=2*self.block_num,
                                            repeat=1))
        
        # [2] Main (conv1d)
        self.block.append(self._conv_block(in_channels=self.channel[2],
                                            out_channels=self.channel[3],
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            stride=self.stride,
                                            n_layers=3*self.block_num,
                                            repeat=1))

        # [3] Main (conv1d)
        self.block.append(self._conv_block(in_channels=self.channel[3],
                                            out_channels=self.channel[4],
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            stride=self.stride,
                                            n_layers=6,
                                            repeat=2))
    

        self.block=nn.Sequential(*self.block)
        

    def lstm_attention_block(self):
        # [6] Attention
        self.block2=nn.Sequential(nn.Linear(self.n_outputs,self.fcchannel[0], bias=False),
                                #   nn.Linear(self.fcchannel[0],self.fcchannel[0]),
                                  nn.Linear(self.fcchannel[0],self.fcchannel[1], bias=False),
                                  nn.Linear(self.fcchannel[1], self.output_shape),
                                  nn.Dropout(0.2)
        )


    def forward(self, xb):
        xb = self.block(xb)   
        xb=xb.flatten(start_dim=1)
        xb=self.block2(xb)        
        
        return xb

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size, padding, stride, n_layers, repeat):
        """
        1-D Convolution output l_out can be defined as
        l_out = (l_in + 2P - (K - 1) - 1) / S + 1,
        where P, K, S denote 'padding', 'kernel_size', and 'stride', respectively.
        """
        modules = nn.ModuleList([])
        
        if repeat==1:
            for i in range(n_layers):
                modules.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm1d(out_channels))
                in_channels=out_channels
        if repeat==2:
            for i in range(n_layers):
                modules.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm1d(out_channels))
                in_channels=out_channels
        modules.append(nn.Dropout(0.25))

        modules.append(nn.MaxPool1d(kernel_size=3, stride=3))
        net = nn.Sequential(*modules)

        return net



if __name__ == '__main__':
    model = ModelArrhythmia(input_shape=[12, 5000], output_shape=9, block_num=1,
                        kernel_size=3, stride=1, padding=1) # CLASS, CHANNEL, TIMEWINDOW
    # pred = model(torch.zeros(50, 1, 20, 250))
    # print(model)

    from pytorch_model_summary import summary
    print(summary(model, torch.zeros((1,12, 5000)), show_input=False))
            # (1, 1, channel, timewindow)
