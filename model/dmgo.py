import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import PAM_Module,MAM_Module

class SPEModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SPEModuleIN, self).__init__()

        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7 ,1 ,1), stride=(3 ,1 ,1), bias=False)
        # self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):

        input = input.unsqueeze(1)

        out = self.s1(input)

        return out.squeeze(1)

class ResSPE(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPE, self).__init__()

        self.spc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(in_channels), )

        self.spc2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True), )

        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))

        return F.leaky_relu(out + input)

class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()

        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(in_channels), )

        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(inplace=True), )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))

        return F.leaky_relu(out + input)

class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(SPAModuleIN, self).__init__()

        # print('k=',k)
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k, 3, 3),  bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        # print(input.size())
        out = self.s1(input)
        out = out.squeeze(2)
        # print(out.size)

        return out

class SPESPA(nn.Module):
    def __init__(self,  k=49):
        super(SPESPA, self).__init__()

        self.layer1 = SPEModuleIN(1, 28)
        self.layer2 = ResSPE(28, 28)
        self.layer3 = ResSPE(28, 28)
        self.layer4 = SPAModuleIN(28, 28, k=k)
        self.bn4 = nn.BatchNorm2d(28)
        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)


    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))  
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn4(F.leaky_relu(x))
        x = self.layer5(x)
        x = self.layer6(x)

        return x

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out

class DMGO(nn.Module):

    def __init__(self, input_channels, input_channels2, n_classes, patch_size):
        super(DMGO, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.planes_a = [128, 64, 28]
        self.planes_b = [8, 16, 32]

        self.conv1_a = SPESPA(k=46)   # Houston k=46 trento k=19
        # For image b (7×7×input_channels2) --> (7×7×planes_b[0])
        self.conv1_b = conv_bn_relu(input_channels2, self.planes_b[0], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[0]) --> (7×7×planes_b[1])
        self.conv2_b = conv_bn_relu(self.planes_b[0], self.planes_b[1], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[1]) --> (7×7×planes_b[2])
        self.conv3_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=0, bias=True)
        self.conv4_b = conv_bn_relu(self.planes_b[2], self.planes_a[2], kernel_size=1, padding=0, bias=True)
        self.PAM_b = PAM_Module(self.planes_a[2])
        self.mam = MAM_Module(self.planes_a[2])

        self.FusionLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.planes_a[2]*2,
                out_channels=self.planes_b[2],
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.planes_b[2]),
            nn.ReLU(),
        )
        self.FusionLayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.planes_a[2] * 2,
                out_channels=self.planes_a[2],
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.planes_a[2]),
            nn.ReLU(),
        )

        self.fc = nn.Linear(self.planes_b[2], n_classes)


    def forward(self, x1, x2):
        # print("X:",x1.shape)
        # print("Y:",x2.shape)
        # input("..............")
        x1 = self.conv1_a(x1)
        # x1 = self.CAM_a(x1)
        x2 = self.conv1_b(x2)
        x2 = self.conv2_b(x2)
        x2 = self.conv3_b(x2)
        x2 = self.conv4_b(x2)
        
        x2 = self.PAM_b(x2)
        x_intr = self.mam(x2,x1)  

        x = torch.cat([x_intr, x2], 1)
        x = self.FusionLayer1(x)
        x = torch.cat([x1, x], 1)
        x_prd = self.FusionLayer1(x)  # x_prd channel:28
        x = self.FusionLayer(x)       # channel: 56 -> 32
        x = self.max_pool(x)
        x_ogm = torch.flatten(x, 1)
        x = self.fc(x_ogm)

        x_intr = torch.mul(x_intr,x_prd)
        logp_intr = F.log_softmax(x_intr, dim=-1)
        p_hsi = F.softmax(x1,dim=-1)
        p_lidar = F.softmax(x2,dim=-1)

        return x, x_ogm, logp_intr, p_hsi, p_lidar
