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



# time:2024/3/25 20:43

这次，将图中表示坐标的d0和表示长度的d1改成了coord和length

统一了在使用过程中，索引和点的id的冲突问题，保留使用了点的id

目前的报错显示，似乎由于你在初始化 pheromone 字典时，没有为所有可能的边添加键值对。需要确保在初始化 pheromone
字典时，为图中的每一条边添加一个键值对。 


# time:2024/3/26 23:12
今日无事

# time:2024/3/28 00:07
暂时解决了在select next node中出现的因为新旧代码数据不匹配导致的keyerror问题。r
然而目前又出现了新的networkx.exception.NetworkXError: The node (596800.6249627918, 2427381.499959573) is not in the
graph.



# time:2024/3/28 13:37
本次将select next node中的节点访问方式从坐标改为了id
不过在update_pheromone中，好像还是有问题

看起来目前代码中对于节点的访问有很多冲突矛盾的部分，这是需要解决的问题

# time:2024/3/29 00:03
全新报错：ValueError: too many values to unpack (expected 2)
原因不明


# time：2024/3/29 22:19
目前看来，在calculate_distance_to_line这个函数中有很大的问题。


# time:2024/3/30 
目前，对于选择的线路的起点终点和初始化信息素时的每条线段的起点终点，有混淆，需要做出区分