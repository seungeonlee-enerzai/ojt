import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Convolution + BatchNormalization + Relu 정의하기
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True): 
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            #layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # 수축 경로(Contracting path)
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)

        # 확장 경로(Expansive path)
        
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.out_layer = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
    
    # forward 함수 정의하기
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)

        dec4_1 = self.dec4_1(enc4_1)

        unpool3 = self.unpool3(dec4_1)

        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.out_layer(dec1_1)

        return x

class VideoRestorationCNN(nn.Module):
    def __init__(self):
        super(VideoRestorationCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class cnn2(nn.Module):
    def __init__(self):
        super(cnn2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        #self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    


class ImageRestorationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=32, num_residual_blocks=12):
        super(ImageRestorationModel, self).__init__()
        print("Call Model 2")
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_features) for _ in range(num_residual_blocks)]
        )

        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)

        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=9, stride=1, padding=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out1 = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += out1
        out = self.conv3(out)
        out = self.sigmoid(out)
        return out
    
class ComplexImageRestorationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_residual_blocks=20):     #저장된 모델은 num_residual_blocks이 16임
        super(ComplexImageRestorationModel, self).__init__()
        print("Call Model 3")
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_features) for _ in range(num_residual_blocks)]
        )

        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=9, stride=1, padding=4)
        #self.bn2 = nn.BatchNorm2d(num_features)

        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=9, stride=1, padding=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out1 = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        out += out1
        out = self.conv3(out)
        out = self.sigmoid(out)
        return out
    
class VComplexImageRestorationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_residual_blocks=20):
        super(VComplexImageRestorationModel, self).__init__()
        print("Call Model 4")
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_features) for _ in range(num_residual_blocks-3)]
        )

        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=9, stride=1, padding=4)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out1 = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += out1
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.sigmoid(out)
        
        return out
    
    


class WidnowImageRestorationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_residual_blocks=16, num_supporting_frame=5):
        super(WidnowImageRestorationModel, self).__init__()
        print("Call Model 3")
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_features) for _ in range(num_residual_blocks)]
        )

        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)

        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=9, stride=1, padding=4)

        self.num_supporting_frame = num_supporting_frame
        self.support_frame_net = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=13, stride=1, padding=6),
            nn.ReLU(),

            ResidualBlock(num_features, num_features),
            ResidualBlock(num_features, num_features),
            ResidualBlock(num_features, num_features),

            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.Conv2d(num_features, num_features, kernel_size=9, stride=1, padding=4)
        )

    def forward_supporting_frames(self, supporting_frames):
        extracted_features = torch.zeros(size=(1, self.conv1.out_channels, 256, 256)).to(supporting_frames[0].device)

        for frame_num in range(self.num_supporting_frame):
            feature = self.support_frame_net(supporting_frames[frame_num])
            extracted_features = extracted_features + feature

        extracted_features = extracted_features#/self.num_supporting_frame
        return extracted_features

    def forward(self, current_frame, supporting_frames):
        out = self.conv1(current_frame)
        out = self.relu(out)
        out1 = out
        support_feature = self.forward_supporting_frames(supporting_frames)
        print(f"Sup: {support_feature.shape}")
        print(f"Out: {out.shape}")
        out += support_feature

        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += out1
        out = self.conv3(out)
        return out