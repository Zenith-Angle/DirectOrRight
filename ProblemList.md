# time: 2024/3/24 11:47
## 目前问题
在第22行，''edge_data[(edge[0], edge[1])] = length # 使用长度作为键''处，edge[0]和edge[1]为什么会=length？
在
```python   
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
```

中，会打印大量的keyerror，并且无法继续运行

目前看起来是```intersection_index```这个有问题


# time: 2024/3/24 17:24
在此前的的数据中，我没有正确的读取到d0这个表示点坐标的属性，现在通过新加了
```python
def read_graphml_with_keys(file_path, keys):
    """
    从 GraphML 文件中读取图数据。

    :param file_path: GraphML 文件的路径
    :param keys: 需要读取的节点或边数据的键名列表
    :return: 一个包含图数据的 NetworkX 图形对象，其中只包含指定键的数据
    """
    # 解析 GraphML 文件为 XML 树结构
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 初始化一个空的无向图
    graph = nx.Graph()

    # 遍历 XML 中的所有节点，将它们添加到图中
    for node in root.iter('{http://graphml.graphdrawing.org/xmlns}node'):
        id = node.attrib['id']  # 节点ID
        data_dict = {}  # 存储节点数据的字典
        for data in node.iter('{http://graphml.graphdrawing.org/xmlns}data'):
            if data.attrib['key'] in keys:  # 如果键在需要读取的键名列表中
                # 收集节点的数据
                data_dict[data.attrib['key']] = data.text
        # 将节点及其数据添加到图中
        graph.add_node(id, **data_dict)

    # 遍历 XML 中的所有边，将它们添加到图中
    for edge in root.iter('{http://graphml.graphdrawing.org/xmlns}edge'):
        source = edge.attrib['source']  # 边的源节点ID
        target = edge.attrib['target']  # 边的目标节点ID
        data_dict = {}  # 存储边数据的字典
        for data in edge.iter('{http://graphml.graphdrawing.org/xmlns}data'):
            if data.attrib['key'] in keys:  # 如果键在需要读取的键名列表中
                # 收集边的数据
                data_dict[data.attrib['key']] = data.text
        # 将边及其数据添加到图中
        graph.add_edge(source, target, **data_dict)

    return graph
```
这个部分完成了读取，并且在Intersection Index中似乎也填充了数据。

目前已有的问题是，似乎有字符串转换的问题，并且报错keyerror:coord
