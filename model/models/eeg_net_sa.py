import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetSA(nn.Module):
    def __init__(self, num_channels=7, winsize=60):
        super(EEGNetSA, self).__init__()

        # Branch block
        padding_max = int(5 * winsize // 4 - 6 // 4)
        self.b_max = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(1, winsize // 2), stride=(1, 3),
                               padding=(0, padding_max))
        self.bn_b_max = nn.BatchNorm2d(64)
        self.relu_b_max = nn.ReLU()

        padding = winsize
        self.b_min = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(1, 3), stride=(1, 3),
                               padding=(0, winsize))
        self.bn_b_min = nn.BatchNorm2d(64)
        self.relu_b_min = nn.ReLU()

        # Depthwise
        self.depthwise = nn.Conv2d(in_channels=128, out_channels=256,
                                   kernel_size=(num_channels, 1),
                                   groups=128, bias=False)
        self.bn_1 = nn.BatchNorm2d(256)
        self.elu_1 = nn.ELU()
        self.ap_1 = nn.AvgPool2d((1, 4))

        # Spatial
        self.spatial_a = nn.Conv2d(in_channels=256, out_channels=256,
                                   kernel_size=(1, 16), groups=256, bias=False,
                                   padding="same")
        self.spatial_b = nn.Conv2d(in_channels=256, out_channels=128,
                                   kernel_size=1, stride=1, padding="same",
                                   dilation=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(128)
        self.elu_2 = nn.ELU()
        self.ap_2 = nn.AvgPool2d((1, 8))

        self.f = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.g = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.h = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)

        self.fc = nn.Linear(64, 1)


    def forward(self, x):
        n = len(x)

        x = torch.transpose(x, -1, -2)       # batch_size * winsize * num_channels -> batch_size * num_channels * winsize
        x = torch.unsqueeze(x, 1)            # -> batch_size * 1 * num_channels * winsize
        
        b_max = self.b_max(x)
        b_max = self.bn_b_max(b_max)
        b_max = self.relu_b_max(b_max)

        b_min = self.b_min(x)
        b_min = self.bn_b_min(b_min)
        b_min = self.relu_b_min(b_min)

        b = torch.cat([b_max, b_min], axis=1)

        b = self.depthwise(b)
        b = self.bn_1(b)
        m_1 = self.elu_1(b)
        ap_1 = self.ap_1(m_1)

        ap_1 = self.spatial_a(ap_1)
        ap_1 = self.spatial_b(ap_1)
        ap_1 = self.bn_2(ap_1)
        m_2 = self.elu_2(ap_1)
        ap_2 = self.ap_2(m_2)

        ap_2 = torch.reshape(ap_2, (n, 128, -1))
        
        k = self.f(ap_2)                                    # n * 64 * -1
        q = torch.reshape(self.g(ap_2), (n, -1, 64))        # n * -1 * 64
        v = torch.reshape(self.h(ap_2), (n, -1, 64))

        qk = torch.bmm(q, k)                                # n * -1 * -1
        qk = qk / (64 ** 0.5)
        qk = torch.softmax(qk, 1)
        z = torch.bmm(qk, v)                                # n * -1 * 64

        z = torch.mean(z, 1)

        out = self.fc(z)
        out = torch.squeeze(out)
        return out