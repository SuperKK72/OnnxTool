import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
from collections import OrderedDict
import cv2
from onnx import shape_inference
logging.basicConfig(level=logging.INFO)
from onnx import shape_inference, TensorProto, version_converter, numpy_helper
logger = logging.getLogger("[ONNXOPTIMIZER]")

def onnx_debug(model):
    logger.info("Test model by onnxruntime")

    # input_shape = model.graph.input[0].type.tensor_type.shape.dim
    #
    # image_shape = [x.dim_value for x in input_shape]
    # image_shape_new = []
    # for x in image_shape:
    #     if x == 0:
    #         image_shape_new.append(1)
    #     else:
    #         image_shape_new.append(x)
    # image_shape = image_shape_new
    # img_array = np.array(np.random.random(image_shape), dtype = np.float32)
    # img = img_array
    image_folder = "./frame_32"
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            image_paths.append(os.path.join(root, file))
    # print(image_paths)
    # exit()
    img_ori = cv2.imread(image_paths[0], -1)
    img_ori = cv2.resize(img_ori, (16, 16))
    img_ori = np.array(img_ori).astype(np.float32)
    img_ori = img_ori.transpose((2, 0, 1))
    img_ori = np.expand_dims(img_ori, 1)

    img_num = len(image_paths)
    img_num = 32
    for i in range(img_num):
        if i == 0:
            continue
        else:
            img = cv2.imread(image_paths[i], -1)
            img = cv2.resize(img, (16, 16))
            img = np.array(img).astype(np.float32)
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, 1)
            img_ori = np.concatenate((img_ori, img), axis=1)
    img = np.expand_dims(img_ori, 0)

    print(model.graph.output[0].name)
    # ori_output = model.graph.output
    ori_output = copy.deepcopy(model.graph.output)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    ort_inputs = {}
    for i, input_ele in enumerate(ort_session.get_inputs()):
        ort_inputs[input_ele.name] = img

    outputs = [x.name for x in ort_session.get_outputs()]
    output_names = outputs[1:]
    print(outputs)
    ort_outs = ort_session.run(outputs, ort_inputs)
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    logger.info("Test model by onnxruntime success")
    del model.graph.output[:]
    model.graph.output.extend(ori_output)
    print(model.graph.output[0].name)
    return ort_outs, output_names
if __name__ == "__main__":
    print("start...")
    onnx_model = onnx.load("./modified_3dconv_ini_16x16/modified_3dconv_ini_16x16.onnx")
    print("load model ok")
    ort_outs, output_names = onnx_debug(onnx_model)
    print("get all tensors ok")
    print("end...")
    print(len(ort_outs))
    print(output_names)
    for l_name in output_names:
        result = ort_outs[l_name]
        # print(type(l_output))
        output_path = "./3d_xwl_modified_16x16/" + l_name + ".txt"
        result = result.transpose((0, 2, 1, 3, 4))
        result = result.flatten()
        output_file = open(output_path, 'w')
        for val in result:
            output_file.write(str('{:.6f}'.format(val)) + '\n')
        output_file.close()