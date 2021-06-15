import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import cv2
import os


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()
        
        self.conv0 = convbn_3d(3, inplanes, kernel_size=1, stride=1, pad=0)

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x):
        inp  = self.conv0(x)
        out  = self.conv6(inp)  #in:1/8 out:1/4

        return out


if __name__ == "__main__":
	# x = torch.rand(1, 3, 32, 128, 128).cuda()
	# net = hourglass(64).cuda()
	# net.eval()
	# with torch.no_grad():
	# 	t1 = time.time()
	# 	for i in range(100):
	# 		out = net(x)
	# 	t2 = time.time()
	# 	print("Execution time: %fs" % (t2 - t1))
	# 	print("Output shape: ")
	# 	print(out.shape)
    x = torch.rand(1, 3, 8, 128, 128)
    net = hourglass(16)
    net.eval()

    image_folder = "./frame_32"
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            image_paths.append(os.path.join(root, file))
    # print(image_paths)

    img_ori = cv2.imread(image_paths[0], -1)
    img_ori = cv2.resize(img_ori, (128, 128))

    img_ori = np.array(img_ori).astype(np.float32)
    img_ori = img_ori.transpose((2, 0, 1))
    img_ori = np.expand_dims(img_ori, 1)

    img_num = len(image_paths)
    img_num = 8
    for i in range(img_num):
        if i == 0:
            continue
        else:
            img = cv2.imread(image_paths[i], -1)
            img = cv2.resize(img, (128, 128))
            img = np.array(img).astype(np.float32)
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, 1)
            img_ori = np.concatenate((img_ori, img), axis=1)
    img_ori = np.expand_dims(img_ori, 0)
    img_ori = torch.Tensor(img_ori)
    result = net.forward(img_ori).detach().numpy()
    print(result.shape)
    result = result.transpose((0, 2, 1, 3, 4)).flatten()

    output_path = os.path.join("./", "3dd_torch", )
    output_path += ".txt"
    output_file = open(output_path, 'w')
    for val in result:
        output_file.write(str('{:.6f}'.format(val)) + '\n')
    output_file.close()

    torch.save(net.state_dict(), "./3dd.pth")

    torch.onnx.export(net, x, "3dd.onnx")

