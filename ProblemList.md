3/24 11:47
## 目前问题
在第22行，''edge_data[(edge[0], edge[1])] = length # 使用长度作为键''处，edge[0]和edge[1]为什么会=length？
在
``python   
try:
if 'coord' in self.graph.nodes[edge[0]] and 'coord' in self.graph.nodes[edge[1]]:
start_key = str(self.graph.nodes[edge[0]]['coord'])
end_key = str(self.graph.nodes[edge[1]]['coord'])
mid_point = self.calculate_mid_point(self.intersection_index[start_key],self.intersection_index[end_key])
else:
print(f"Missing 'coord' for edge {edge}")
continue # 跳过这个边
except KeyError as e:
print(f"KeyError for edge {edge}: {e}")  # 打印出问题的边和错误信息
continue # 跳过这个边
``

中，会打印大量的keyerror，并且无法继续运行

目前看起来是``intersection_index``这个有问题

