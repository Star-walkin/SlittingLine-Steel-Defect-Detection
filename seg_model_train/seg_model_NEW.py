
import torch
import torch.nn as nn





class down_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.act1 = nn.ReLU(inplace=True)

        #self.conv2 =nn.Conv2d(self.out_channels,self.out_channels,kernel_size=1,stride=1,padding=0,groups=self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        self.act2 = nn.ReLU(inplace=True)


    def forward(self, x):

        x= self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))

        return x


class up_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
        #self.conv1 = nn.Conv2d(self.out_channels, self.out_channels,3,1,1,groups=self.out_channels)
        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.act1 = nn.ReLU(inplace=True)

        #self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0,groups=self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        self.act2 = nn.ReLU(inplace=True)


    def forward(self, from_up, from_down):

        from_up = self.upconv(from_up)
        #x = torch.cat((from_up, from_down), 1)
        #x=
        x = self.act1(self.norm1(self.conv1(from_up+from_down)))
        #x = self.conv1(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x


import torch
import torch.nn as nn

# down_layer 和 up_layer 保持不变，只需修改 UNet 类

class UNet(nn.Module):
    # 【修改1】 增加 out_channels 参数，默认值给 1，但在主程序里我们会传 3
    def __init__(self, in_channels=3, out_channels=1, dims=[32, 64, 128, 256, 256]):
        super().__init__()

        # 使用传入的 in_channels
        self.proj = down_layer(in_channels, dims[0], 1, 1, 0)

        self.down1 = down_layer(dims[0], dims[1], 3, 2, 1)
        self.down2 = down_layer(dims[1], dims[2], 3, 2, 1)
        self.down3 = down_layer(dims[2], dims[3], 3, 2, 1)
        self.down4 = down_layer(dims[3], dims[4], 3, 2, 1)

        self.up1 = up_layer(dims[4], dims[3])
        self.up2 = up_layer(dims[3], dims[2])
        self.up3 = up_layer(dims[2], dims[1])
        self.up4 = up_layer(dims[1], dims[0])

        # 【修改2】 使用传入的 out_channels，不再写死 1
        self.out = nn.Sequential(
            nn.Conv2d(dims[0], out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()  # Tanh 对应归一化范围
        )

    def forward(self, x):
        x0 = self.proj(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        x = self.out(x)
        return x



if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import torchvision.transforms.functional as TF
    model = UNet().cuda()
    image = torch.randn(1, 3, 256, 256).cuda()
    #image = Image.open(r"F:\mvtec\bottle\noise_train\mask\0.png").convert("RGB")
    #image = TF.to_tensor(image).unsqueeze(0).cuda()
    y=model(image)
    #'''
    from thop import profile
    print(torch.cuda.is_available())
    # input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, (image,))
    total = sum([param.nelement() for param in model.parameters()])
    print(total / 1e6)
    print('flops: %.2f B, params: %.2f M' % (flops / 1e9, params / 1e6))

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(image)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(image)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)
    #'''
