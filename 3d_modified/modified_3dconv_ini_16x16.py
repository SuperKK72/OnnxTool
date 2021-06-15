import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv0 = convbn_3d(3, inplanes, kernel_size=1, stride=1, pad=0)

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x):
        inp = self.conv0(x)
        out = self.conv1(inp)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16
        post = F.relu(self.conv5(out) + pre, inplace=True)
        out = self.conv6(post)  # in:1/8 out:1/4

        return out


if __name__ == "__main__":
    x = torch.rand(1, 3, 32, 16, 16)
    net = hourglass(64)
    net.eval()
    # with torch.no_grad():
    # 	t1 = time.time()
    # 	for i in range(100):
    # 		out = net(x)
    # 	t2 = time.time()
    # 	print("Execution time: %fs" % (t2 - t1))
    # 	print("Output shape: ")
    # 	print(out.shape)
    torch.save(net.state_dict(), "./modified_3dconv_ini_16x16/modified_3dconv_ini_16x16.pth")
    torch.onnx.export(net, x, "./modified_3dconv_ini_16x16/modified_3dconv_ini_16x16.onnx")
