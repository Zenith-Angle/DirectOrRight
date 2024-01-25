# IntersectionIndexer.py

import networkx as nx


def create_intersection_index(graph):
    """
    为图中的每个节点（路口）创建一个唯一的索引。

    :param graph: 一个networkx图，代表城市的路网。
    :return: 一个字典，键为路口的坐标，值为唯一的索引编号。
    """
    intersection_index = {}
    for i, node in enumerate(graph.nodes):
        intersection_index[node] = i
    return intersection_index


def load_graph(graph_path):
    """
    加载图形数据文件。

    :param graph_path: 图形数据文件的路径。
    :return: networkx图对象。
    """
    return nx.read_graphml(graph_path)


# 示例代码
if __name__ == "__main__":
    # 载入图形数据
    graph_path = 'data/paris_street_network.graphml'  # 替换为实际的文件路径
    G = load_graph(graph_path)

    # 创建路口索引
    intersection_index = create_intersection_index(G)

    # 打印一些样本数据以验证
    sample_index_data = {key: intersection_index[key] for key in list(intersection_index)[:5]}
    print(sample_index_data)
