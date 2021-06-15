import onnx
path = "./3dd/3dd.onnx"
model = onnx.load(path)
graph = model.graph
node = graph.node
find_idx = 0
for i in range(len(node)):
    if node[i].op_type == 'ConvTranspose':
        node_rise = node[i]
        if node_rise.output[0] == '15':
            find_idx = i
            print(i)  # 157
# exit()
# output_info_0 = onnx.helper.make_tensor_value_info("output0",elem_type=1,shape=[1,128,8,32,32])
# output_info_1 = onnx.helper.make_tensor_value_info("output1",elem_type=1,shape=[1,128,16,64,64])
# output_info_2 = onnx.helper.make_tensor_value_info("output2",elem_type=1,shape=[1,128,16,64,64])
output_info = onnx.helper.make_tensor_value_info("output_info",elem_type=1,shape=[1,16,16,256,256])

# node[7].output.append("output0")
new_node = node[:2]
del model.graph.node[:]
model.graph.node.extend(new_node)
model.graph.node[0].input[0] = "0"
model.graph.node[1].output[0] = "output_info"
old_output = model.graph.output[0]
model.graph.output.remove(old_output)
model.graph.output.insert(0,output_info)








onnx.checker.check_model(model)
onnx.save(model, './3dd/3dd_revision1.onnx')