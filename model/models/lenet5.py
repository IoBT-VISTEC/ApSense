import torch
import torch.nn as nn
import torch.nn.functional as F

## Modified LeNet 5 
## Modify output from (2 with Softmax to 1 without Softmax
class ModifiedLeNet5(nn.Module):
    def __init__(self,num_channels=7, winsize=60, weight=1e-3):
        super(ModifiedLeNet5, self).__init__()
        weight = 1e-3
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        
        self.dropout = nn.Dropout(p=0.8)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_conv_output([num_channels,winsize]), 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.weight = weight

        # Apply weight initialization
        self._initialize_weights()

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, shape[0],shape[1])
            output_feat = self._forward_features(input)
            n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))

        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, self.weight)
                nn.init.constant_(m.bias, self.weight)

    def forward(self, x):
        x = torch.transpose(x, -1, -2)   # batch_size * winsize * num_channels -> batch_size * num_channels * winsize
        x = self._forward_features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.squeeze(x)
        return x


