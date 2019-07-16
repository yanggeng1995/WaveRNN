import numpy as np
import torch
import torch.nn as nn
from layers.wavernn import WaveRNN
from layers.modules import UpsampleNet, FrameRateNet

class Model(nn.Module):
    def __init__(self,
                 quantization_channels=256,
                 gru_channels=896,
                 fc_channels=896,
                 lc_channels=80,
                 lc_out_channles=80,
                 upsample_factor=(5, 5, 8),
                 use_lstm=True,
                 lstm_layer=2,
                 upsample_method='duplicate'):
        super().__init__()
        self.frame_net = FrameRateNet(lc_channels, lc_out_channles)
        self.upsample = UpsampleNet(input_size=lc_out_channles,
                output_size=lc_out_channles,
                upsample_factor=upsample_factor,
                use_lstm=use_lstm,
                lstm_layer=lstm_layer,
                upsample_method=upsample_method)
        self.wavernn = WaveRNN(quantization_channels, gru_channels, fc_channels, lc_channels)
        self.num_params()

    def forward(self, inputs, conditions):
        conditions = self.frame_net(conditions.transpose(1, 2))
        conditions = self.upsample(conditions.transpose(1, 2))
        return self.wavernn(inputs, conditions[:, 1:, :])

    def after_update(self):
        self.wavernn.after_update()

    def generate(self, conditions):
        self.eval()
        with torch.no_grad():
            conditions = self.frame_net(conditions.transpose(1, 2))
            conditions = self.upsample(conditions.transpose(1, 2))
            output = self.wavernn.generate(conditions)
        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)
