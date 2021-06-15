import onnx
from onnx import shape_inference

path = "./yolov5_kk.onnx"
model = onnx.load(path)
graph = model.graph
# print(graph.node)
node = graph.node
# print(node)
# graph.node[0].input[0]="999"
# reorg_node = onnx.helper.make_node("SpaceToDepth",inputs=["inputkk"],outputs=["999"],name="reorg_kk",blocksize=2)
# graph.node.insert(0, reorg_node)


# for i in range(len(node)):
#     if node[i].op_type == 'Conv':
#         node_rise = node[i]
#         if node_rise.output[0] == '168':
#             find_idx = i
#             print(i)  # 157

#
# remove_node = 0
# for id in ['381','420','459']:
#     for i in range(len(node)):
#         if node[i].op_type == 'Reshape':
#             node_rise = node[i]
#             if node_rise.output[0] == id:
#                 remove_node = i
#                 print(i)  # 157
#     graph.node.remove(node[remove_node])

# node[0].input[0] = "inputkk"
# print(node[41].input)
# find_idx = 0
# for i in range(len(node)):
#     if node[i].op_type == 'Conv':
#         node_rise = node[i]
#         if node_rise.output[0] == '456':
#             find_idx = i
#             print(i)  # 157
# node[find_idx].input[0]="inputkk"

old_input = graph.input[0]
input_info = onnx.helper.make_tensor_value_info("inputkk",elem_type=1,shape=[1,3,640,640])
graph.input.remove(old_input)
graph.input.insert(0,input_info)

# print(graph.output)
# print(node[find_idx].output)
# output_info0 = onnx.helper.make_tensor_value_info("output0",elem_type=1,shape=[1,255,80,80])
# output_info1 = onnx.helper.make_tensor_value_info("output1",elem_type=1,shape=[1,255,40,40])
# output_info2 = onnx.helper.make_tensor_value_info("output2",elem_type=1,shape=[1,255,20,20])
# node[204].output[0] = "output0"
# node[205].output[0] = "output1"
# node[206].output[0] = "output2"
# node[241].output.insert(1,"output0")
# node[278].output.insert(1,"output1")
# node[204].output.insert(1,"output2")


# for i in range(2):
#     graph.output.remove(graph.output[0])
# print(graph.output)

# graph.output.insert(0,output_info2)
# graph.output.insert(0,output_info1)
# graph.output.insert(0,output_info0)


# new_nodes = node[:205]
# new_nodes.append(node[241])
# new_nodes.append(node[278])
#
# del model.graph.node[:]
# model.graph.node.extend(new_nodes)
# model.graph.node[0].input[0] = "inputkk"
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, 'yolov5_kk.onnx')