import torch
import torch.nn as nn

class VGG13_net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # The input data is 1x256x130
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, (3, 3), 1, 1),            # Output = 24*256*130
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, (3, 3), 1, 1),           # Output = 24*256*130
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # Output = 24*128*65
            nn.Conv2d(24, 48, (3, 3), 1, 1),           # Output = 48*128*65
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, (3, 3), 1, 1),           # Output = 48*128*65
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # Output = 48*64*32
            nn.Conv2d(48, 96, (3, 3), 1, 1),           # Output = 96*64*32
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, (3, 3), 1, 1),           # Output = 96*64*32
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # Output = 96*32*16
            nn.Conv2d(96, 192, (3, 3), 1, 1),          # Output = 192*32*16
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, (3, 3), 1, 1),         # Output = 192*32*16
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # Output = 192*16*8
            nn.Conv2d(192, 192, (3, 3), 1, 1),         # Output = 192*16*8
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, (3, 3), 1, 1),         # Output = 192*16*8
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # Output = 192*8*4
        )
        
        # Final feature map = 192 * 8 * 4 = 6144
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 8 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
