import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor,
                 use_lstm=True,
                 lstm_layer=2,
                 upsample_method="duplicate"):

        super(UpsampleNet, self).__init__()
        self.upsample_method = upsample_method
        self.upsample_factor = upsample_factor
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm_layer = nn.LSTM(input_size, output_size, num_layers=lstm_layer, batch_first=True)
        if upsample_method == 'duplicate':
            self.upsample_factor = int(np.prod(upsample_factor))
        elif upsample_method == 'transposed_conv2d':
            assert isinstance(upsample_factor, tuple)
            kernel_size = 3
            self.upsamples = nn.ModuleList()
            for u in upsample_factor:
                padding = (kernel_size - 1) // 2
                conv = nn.ConvTranspose2d(1, 1, (kernel_size, 2 * u),
                                          padding=(padding, u // 2),
                                          dilation=1, stride=(1, u))
                self.upsamples.append(conv)

    def forward(self, inputs):
        if self.use_lstm:
           inputs, _ = self.lstm_layer(inputs.transpose(1, 2))
           inputs = inputs.transpose(1, 2)
        if self.upsample_method == 'duplicate':
            output = F.interpolate(inputs, scale_factor=self.upsample_factor, mode='nearest')
        elif self.upsample_method == 'transposed_conv2d':
            output = input.unsqueeze(1)
            for layer in self.upsamples:
                output = layer(output)
            output = output.squeeze(1)
            output = output[:, :, : input.size(-1) * np.prod(self.upsample_factor)]

        return output.transpose(1, 2)

class FrameRateNet(nn.Module):

    def __init__(self,
                 lc_channels=30,
                 out_channels=256):

        super(FrameRateNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(lc_channels, lc_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(lc_channels, lc_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.fc = nn.Sequential(
            nn.Linear(lc_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels),
            nn.Tanh()
        )

    def forward(self, condition):

        residual = self.convs(condition)
        output = residual + condition
        output = self.fc(output.transpose(1, 2))
        return output
