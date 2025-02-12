import torch
import torch.nn as nn


from .convblocks import get_conv_block
from .feed_forward import LinearFeedFwd, LocalizationFeedFwd
from torch import Tensor
from typing import List, Optional

def linear_softmax_pooling(x: Tensor):
    return (x ** 2).sum(1) / x.sum(1)


def weighted_sum_pooling(x: Tensor, w: Tensor):
    w = torch.clip(w, 1e-7, 1.)
    return x.sum(1) / w.sum(1)
class Crnn(nn.Module):
    def __init__(
            self,
            num_freq: int,
            num_class: int,
            conv_block: str = 'conv',
            n_channels: List[int] = [16, 32, 64, 128, 128],
            pooling_sizes: List[int] = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)],            
            gru_layers: int = 2,
            dropout: float = 0.0,
            gru_hidden: Optional[int] = None):
        ##############################
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        super().__init__()

        # if len(pooling_sizes) != 3:
        #     raise ValueError('pooling_sizes should be a list of 3 ints')
        # if len(n_channels) != 3:
        #     raise ValueError('n_channels should be a list of 3 ints')

        self.bn = nn.BatchNorm2d(1)

        conv_block = get_conv_block(conv_block)
        self.conv1 = conv_block(
            1, n_channels[0], pooling_size=pooling_sizes[0])
        self.conv2 = conv_block(
            n_channels[0], n_channels[1], pooling_size=pooling_sizes[1])
        self.conv3 = conv_block(
            n_channels[1], n_channels[2], pooling_size=pooling_sizes[2])
        self.conv4 = conv_block(
            n_channels[2], n_channels[3], pooling_size=pooling_sizes[3]
        )
        self.conv5 = conv_block(
            n_channels[3], n_channels[4], pooling_size=pooling_sizes[4]
        )
        h_factor = num_freq
        for psz in pooling_sizes:
            h_factor //= psz[-1]
            print('h factor', h_factor)

        # hid_size = n_channels[-1] * h_factor
        hid_size = n_channels[-1]*2
        print('hfactor', h_factor)
        if gru_hidden is None:
            gru_hidden = hid_size
        print('hid_size', hid_size, 'gru_hidden', gru_hidden)
        self.gru = nn.GRU(
            hid_size, gru_hidden, gru_layers, dropout=dropout,
            batch_first=True, bidirectional=True)

        self.ffwd = LinearFeedFwd(gru_hidden * 2, num_class)

    def detection(self, x):
        ##############################
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = self.bn(x.unsqueeze(1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # B, C, T, F
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.permute(0, 2, 1, 3)  # B, T, C, F
        x = x.flatten(start_dim=2)  # B, T, C * F
        print(x.shape)
        x = self.gru(x)[0]  # B, T, C * F
        x = self.ffwd(x)  # B, T, class_num
        return torch.sigmoid(x)

    def forward(self, x):
        frame_wise_prob = self.detection(x)  # B, T, ncls
        clip_prob = linear_softmax_pooling(frame_wise_prob)  # B, ncls
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob,
            'frame_prob': frame_wise_prob
        }
