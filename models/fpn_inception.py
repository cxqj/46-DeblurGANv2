import torch
import torch.nn as nn
from pretrainedmodels import inceptionresnetv2
# from torchsummary import summary
import torch.nn.functional as F

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_in, num_out, norm_layer):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(num_in, num_out, kernel_size=3, padding=1),
                                 norm_layer(num_out),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class FPNInception(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):   # norm_layer = instance
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer)

        # The segmentation heads on top of the FPN
        """
          FPNHead(
              (block0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (block1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            )
        """
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        """
          FPNHead(
              (block0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (block1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            )
        """
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        """
          FPNHead(
              (block0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (block1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            )
        """
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        """
          FPNHead(
              (block0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (block1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            )
        """
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        """
          Sequential(
              (0): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
              (2): ReLU()
            )
        """
        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        """
          Sequential(
              (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
              (2): ReLU()
            )
        """
        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        """
          Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        """
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):  # x : (1,3,256,256)
        """
          map0: (1,128,128,128)
          map1: (1,256,64,64)
          map2: (1,256,32,32)
          map3: (1,256,16,16)
          map4: (1,256,8,8)
        """
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest") # (1,128,64,64)
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest") # (1,128,64,64)
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest") # (1,128,64,64)
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest") # (1,128,64,64)

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))  # (1,512,64,64)-->(1,128,64,64)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")  # (1,128,128,128)
        smoothed = self.smooth2(smoothed + map0)  # (1,64,128,128)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")  # (1,64,256,256)

        final = self.final(smoothed)  # (1,3,256,256)
        res = torch.tanh(final) + x  # (1,3,256,256)

        return torch.clamp(res, min = -1,max = 1)


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=256):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        """
        enc0(
              (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU()
            )
        """
        self.enc0 = self.inception.conv2d_1a  #3-->32
        """
        enc1:Sequential(
                  (0): BasicConv2d(
                    (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (1): BasicConv2d(
                    (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
                )
        """
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,  # 32-->32
            self.inception.conv2d_2b,  # 32-->64
            self.inception.maxpool_3a,
        ) # 64
        """
        enc2:Sequential(
              (0): BasicConv2d(
                (conv): Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (1): BasicConv2d(
                (conv): Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
        """
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,  # 64-->80
            self.inception.conv2d_4a,  # 80-->192
            self.inception.maxpool_5a,
        )  # 192
        """
        Sequential(
              (0): Mixed_5b(
                (branch0): BasicConv2d(
                  (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (branch1): Sequential(
                  (0): BasicConv2d(
                    (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (1): BasicConv2d(
                    (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
                    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                )
                (branch2): Sequential(
                  (0): BasicConv2d(
                    (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (1): BasicConv2d(
                    (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (2): BasicConv2d(
                    (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                )
                (branch3): Sequential(
                  (0): AvgPool2d(kernel_size=3, stride=1, padding=1)
                  (1): BasicConv2d(
                    (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                )
              )
              (1): Sequential(
                (0): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (1): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (2): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (3): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (4): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (5): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (6): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (7): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (8): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
                (9): Block35(
                  (branch0): BasicConv2d(
                    (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (branch1): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (branch2): Sequential(
                    (0): BasicConv2d(
                      (conv): Conv2d(320, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (1): BasicConv2d(
                      (conv): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                    (2): BasicConv2d(
                      (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                      (relu): ReLU()
                    )
                  )
                  (conv2d): Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1))
                  (relu): ReLU()
                )
              )
              (2): Mixed_6a(
                (branch0): BasicConv2d(
                  (conv): Conv2d(320, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
                  (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (branch1): Sequential(
                  (0): BasicConv2d(
                    (conv): Conv2d(320, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (1): BasicConv2d(
                    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                  (2): BasicConv2d(
                    (conv): Conv2d(256, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
                    (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    (relu): ReLU()
                  )
                )
                (branch2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
              )
            )
        """
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )   # 1088

        """
                   Sequential(
          (0): Sequential(
            (0): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (1): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (2): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (3): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (4): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (5): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (6): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (7): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (8): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (9): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (10): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (11): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (12): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (13): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (14): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (15): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (16): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (17): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (18): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
            (19): Block17(
              (branch0): BasicConv2d(
                (conv): Conv2d(1088, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (branch1): Sequential(
                (0): BasicConv2d(
                  (conv): Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (1): BasicConv2d(
                  (conv): Conv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
                  (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
                (2): BasicConv2d(
                  (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
                  (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU()
                )
              )
              (conv2d): Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1))
              (relu): ReLU()
            )
          )
          (1): Mixed_7a(
            (branch0): Sequential(
              (0): BasicConv2d(
                (conv): Conv2d(1088, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (1): BasicConv2d(
                (conv): Conv2d(256, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
            )
            (branch1): Sequential(
              (0): BasicConv2d(
                (conv): Conv2d(1088, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (1): BasicConv2d(
                (conv): Conv2d(256, 288, kernel_size=(3, 3), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
            )
            (branch2): Sequential(
              (0): BasicConv2d(
                (conv): Conv2d(1088, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (1): BasicConv2d(
                (conv): Conv2d(256, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
              (2): BasicConv2d(
                (conv): Conv2d(288, 320, kernel_size=(3, 3), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU()
              )
            )
            (branch3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
        )
        """
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        ) #2080
        """
        Sequential(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
        """
        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))  # (256,256)
        """
         Sequential(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
        """
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        """
        Sequential(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
        """
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        """
         ReflectionPad2d((1, 1, 1, 1))
        """
        self.pad = nn.ReflectionPad2d(1)
        """
         Conv2d(2080, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        """
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        """
         Conv2d(1088, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        """
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        """
          Conv2d(192, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        """
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        """
          Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        """
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        """
          Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        """
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):  # (1,3,256,256)

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)    # (1,32,127,127)

        enc1 = self.enc1(enc0) # (1,64,62,62)

        enc2 = self.enc2(enc1) # (1,192,29,29)

        enc3 = self.enc3(enc2) # (1,1088,14,14)

        enc4 = self.enc4(enc3) # (1,2080,6,6)

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))  # (1,2088,6,6)-->(1,256,8,8)
        lateral3 = self.pad(self.lateral3(enc3))  # (1,1088,14,14)-->(1,256,16,16)
        lateral2 = self.lateral2(enc2)   # (1,192,29,29)-->(1,256,29,29)
        lateral1 = self.pad(self.lateral1(enc1))  # (1,64,62,62)-->(1,256,64,64)
        lateral0 = self.lateral0(enc0)  # (1,32,127,127)-->(1,128,127,127)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4  # (1,256,8,8)
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))  # (1,256,16,16) + [(1,256,8,8)-->(1,256,16,16)] -- > (1,256,16,16)
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))  # (1,256,32,32)
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))  # (1,256,64,64)
        return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4



