import onnx
import onnxruntime as rt
import cv2
import os
import numpy as np

# a = [[1,1,1],[1,1,1]]
# b = [[2,2,2],[2,2,2]]
# c = np.concatenate((a,b),axis=0)
# print(c)
# print(c.shape)
# exit()



model_path = "./3d_revision2.onnx"
model = onnx.load(model_path)
m = rt.InferenceSession(model_path)

image_folder = "./frame_32"
image_paths = []
for root, dirs, files in os.walk(image_folder):
    for file in files:
        image_paths.append(os.path.join(root, file))
# print(image_paths)
# exit()
img_ori = cv2.imread(image_paths[0], -1)
img_ori = cv2.resize(img_ori,(128,128))
img_ori = np.array(img_ori).astype(np.float32)
img_ori = img_ori.transpose((2,0,1))
img_ori = np.expand_dims(img_ori, 1)
    
img_num = len(image_paths)
for i in range(img_num):
    if i == 0:
        continue
    else:
        img = cv2.imread(image_paths[i], -1)
        img = cv2.resize(img,(128,128))
        img = np.array(img).astype(np.float32)
        img = img.transpose((2,0,1))
        img = np.expand_dims(img, 1)
        img_ori = np.concatenate((img_ori, img), axis=1)
img_ori = np.expand_dims(img_ori, 0)

input_name = m.get_inputs()[0].name
output_name = m.get_outputs()[0].name
result = m.run([output_name], {input_name: img_ori})
result = np.array(result[0])
result = result.transpose((0,2,1,3,4))
print(result.shape)
result = result.flatten()

output_path = os.path.join("./",output_name,)
output_path += ".txt"
output_file = open(output_path, 'w')
for val in result:
    output_file.write(str('{:.6f}'.format(val))+'\n')
output_file.close()



