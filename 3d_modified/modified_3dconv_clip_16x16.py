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

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x):
        inp = self.conv0(x)

        out = self.conv6(inp)  # in:1/8 out:1/4

        return out


if __name__ == "__main__":
    x = torch.rand(1, 3, 4, 4, 4)
    net = hourglass(8)
    net.eval()
    # with torch.no_grad():
    # 	t1 = time.time()
    # 	for i in range(100):
    # 		out = net(x)
    # 	t2 = time.time()
    # 	print("Execution time: %fs" % (t2 - t1))
    # 	print("Output shape: ")
    # 	print(out.shape)
    torch.save(net.state_dict(), "./modified_3dconv_clip_16x16/modified_3dconv_clip_16x16.pth")
    torch.onnx.export(net, x, "./modified_3dconv_clip_16x16/modified_3dconv_clip_16x16.onnx")
