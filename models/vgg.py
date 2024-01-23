import torch.nn as nn

'''
The given VGG-style architecture is defined by a configuration list (cfg) that specifies the number and size of convolutional layers and when to apply max pooling layers.
The list cfg specifies layer types and channel dimensions for the convolutional layers. Here's a breakdown:

The numbers represent convolutional layers with the corresponding number of output channels.
'M' represents a max pooling layer.
'A' represents another max pooling layer with different kernel size and stride.
From the cfg list, we can count the layers:

There are 8 convolutional layers (represented by the numbers 64, 128, 256, and 512, each appearing twice).
There are 3 max pooling layers represented by 'M'.
There is 1 additional max pooling layer represented by 'A'.
In addition to these, the self.classifier contains two fully connected layers (represented by nn.Linear(512, 512) and nn.Linear(512, nclasses)), along with batch normalization, a ReLU activation, and dropout.

So in total:

8 convolutional layers
4 max pooling layers
2 fully connected layers
This gives us a total of 14 layers that have learnable parameters (all convolutional and fully connected layers). The pooling layers, batch normalization, ReLU activations, and dropout do not add to the count of layers in terms of depth, as they do not have learnable parameters that affect the depth of the architecture in the same way that convolutional and fully connected layers do.
'''

class Network(nn.Module):
    def __init__(self, nchannels, nclasses):
        super(Network, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'A']
        self.features = make_layers(cfg, nchannels)
        self.classifier = nn.Sequential( nn.Linear( 512, 512 ), nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear( 512, nclasses))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.5)]
        elif v == 'A':
            layers += [nn.MaxPool2d(kernel_size=4, stride=4), nn.Dropout(0.5)]
        else: #nn.BatchNorm2d(v)
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
