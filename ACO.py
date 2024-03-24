# ACO.py
import math
import random
import networkx as nx
from TimeRecorder import TimeRecorder
from IntersectionIndexer import create_intersection_index
from tqdm import tqdm  # 添加tqdm库
import xml.etree.ElementTree as ET


def create_intersection_index(graph):
    """
    创建交叉点索引和边长数据。

    参数:
    - graph: 包含节点和边的数据结构，节点需包含'd0'属性表示坐标，边需包含'd1'属性表示长度。

    返回:
    - intersection_index: 字典，键为交叉点的坐标字符串，值为交叉点的坐标。
    - edge_data: 字典，键为边的节点对，值为边的长度。
    """
    intersection_index = {}
    edge_data = {}

    # 为每个具有'd0'属性的节点创建交叉点索引
    for node, data in graph.nodes(data=True):
        if 'd0' in data:
            print(f"Original 'd0' value for node {node}: {data['d0']}")  # 打印原始的 'd0' 值
            coords = data['d0'].replace('(', '').replace(')', '').split(',')
            print(f"Split 'd0' value for node {node}: {coords}")  # 打印分割后的 'd0' 值
            try:
                coord_tuple = (float(coords[0]), float(coords[1]))
                print(f"Converted 'd0' value for node {node}: {coord_tuple}")  # 打印转换后的 'd0' 值
                intersection_index[str(coord_tuple)] = coord_tuple
            except ValueError as e:
                print(f"Error converting 'd0' value for node {node}: {e}")  # 打印转换错误
        else:
            print(f"Node {node} does not have 'd0' attribute")  # 打印没有 'd0' 属性的节点

    # 打印交叉点索引
    print("Intersection Index: ", intersection_index)

    # 为每条具有'd1'属性的边记录边长
    for edge in graph.edges(data=True):
        if 'd1' in edge[2]:
            length = edge[2]['d1']
            edge_data[(edge[0], edge[1])] = length

    return intersection_index, edge_data


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



class AntColonyOptimizer:
    def __init__(self, graph, num_ants, evaporation_rate, iterations):
        """
        初始化蚁群优化算法。
        """
        self.graph = graph
        self.num_ants = num_ants
        self.evaporation_rate = 0.15  # 降低蒸发率到 0.15
        self.iterations = iterations
        self.time_recorder = TimeRecorder()
        self.intersection_index, self.edge_data = create_intersection_index(graph)
        # 初始化信息素，确保边的每个节点都使用了正确的索引
        self.pheromone = {}

        for edge in self.graph.edges():
            if 'coord' in self.graph.nodes[edge[0]] and 'coord' in self.graph.nodes[edge[1]]:
                start_key = str(self.graph.nodes[edge[0]]['coord'])
                end_key = str(self.graph.nodes[edge[1]]['coord'])
                mid_point = self.calculate_mid_point(self.intersection_index[start_key],
                                                     self.intersection_index[end_key])
            else:
                continue  # 跳过这个边

    def select_start_end_points(self):
        """
        随机选择起点和终点。
        """
        nodes = list(self.graph.nodes())
        start = random.choice(nodes)
        end = random.choice(nodes)
        while end == start:
            end = random.choice(nodes)

        # 确保start和end在intersection_index中对应的值是坐标列表或元组
        # 添加检查以确保start和end作为键存在于self.intersection_index中
        start_key = str(self.graph.nodes[start]['coord'])
        end_key = str(self.graph.nodes[end]['coord'])
        if start_key not in self.intersection_index or end_key not in self.intersection_index:
            raise KeyError(
                f"One or both of the selected points {start_key} or {end_key} do not exist in intersection_index.")

        if not (isinstance(self.intersection_index[start_key], (list, tuple))
                and isinstance(self.intersection_index[end_key], (list, tuple))):
            raise ValueError("Intersection index values for selected nodes must be lists or tuples.")

        return self.intersection_index[start_key], self.intersection_index[end_key]

    def initialize_pheromone(self):
        """
        在起点和终点连线上初始化更多信息素。
        """
        start, end = self.select_start_end_points()  # 获取起点和终点坐标
        for edge in self.graph.edges():
            mid_point = self.calculate_mid_point(self.intersection_index[edge[0]], self.intersection_index[edge[1]])
            distance = self.calculate_distance_to_line(start, end, mid_point)
            self.pheromone[edge] *= (1 / (1 + distance))  # 距离越短，信息素量越大

    def calculate_mid_point(self, start, end):
        """
        计算两点的中点。
        """
        if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple))):
            raise TypeError("start and end must be lists or tuples.")

        return ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)

    @staticmethod
    def calculate_distance_to_line(start, end, point):
        """
        计算点到直线的距离。
        """
        x0, y0 = point
        x1, y1 = start
        x2, y2 = end
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        return num / den

    def calculate_angle(self, node1, node2, node3):
        """
        计算由三个节点形成的两条道路之间的夹角。

        :param node1: 第一个节点的坐标。
        :param node2: 第二个节点的坐标（转弯点）。
        :param node3: 第三个节点的坐标。
        :return: 两条道路之间的夹角（以度为单位）。
        """

        def angle_between_points(p1, p2, p3):
            a = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            b = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
            angle = math.degrees(a - b)
            return angle if angle >= 0 else angle + 360

        return angle_between_points(node1, node2, node3)

    def get_previous_node(self, current_node, path):
        """
        获取与当前节点相连的上一个节点。

        :param current_node: 当前节点。
        :param path: 蚂蚁走过的路径。
        :return: 上一个节点。
        """
        current_index = path.index(current_node)
        if current_index > 0:
            return path[current_index - 1]
        else:
            return None  # 如果当前节点是路径的起点，则没有上一个节点

    def can_turn(self, current_node_index, next_node_index, traffic_mode, path):
        """
        根据交通规则判断是否可以从当前节点转向下一个节点。

        :param current_node_index: 当前节点的索引。
        :param next_node_index: 下一个节点的索引。
        :param traffic_mode: 交通模式，1代表自由转弯，2代表限制转弯。
        :param path: 蚂蚁走过的路径。
        :return: 是否可以转向。
        """
        if traffic_mode == 1:  # 自由转弯模式
            return True
        elif traffic_mode == 2:  # 限制转弯模式
            # 获取上一个节点的索引
            prev_node_index = self.get_previous_node(current_node_index, path)
            if prev_node_index is None:
                return True  # 如果没有上一个节点，说明这是路径的起点，可以自由转弯

            # 确认节点存在并且有'coord'属性
            if prev_node_index not in self.graph.nodes or 'coord' not in self.graph.nodes[prev_node_index]:
                print(f"Error: Node {prev_node_index} not found or 'coord' attribute missing.")
            return False  # 或根据您的需求进行其他错误处理

            # 获取节点的坐标
            prev_coord = self.graph.nodes[prev_node_index]['coord']
            curr_coord = self.graph.nodes[current_node_index]['coord']
            next_coord = self.graph.nodes[next_node_index]['coord']

            # 计算夹角
            angle = self.calculate_angle(prev_coord, curr_coord, next_coord)

            # 判断是否是直行或右转
            return 160 <= angle <= 180 or 80 <= angle <= 100  # 这里的角度范围可以根据实际情况调整

    def find_path(self, start_index, end_index, traffic_mode):
        """
        对于每只蚂蚁寻找从起点到终点的路径。

        :param start_index: 路径的起点索引。
        :param end_index: 路径的终点索引。
        :param traffic_mode: 交通模式。
        :return: 路径列表和总时长。
        """
        path = [start_index]  # 使用起点索引初始化路径
        current_node = list(self.graph.nodes())[start_index]  # 当前节点为起始节点

        total_duration = 0  # 总时长，包括行驶时间和路口等待时间
        max_iterations = 100  # 设置最大迭代次数以避免无限循环

        for _ in range(max_iterations):
            next_node, duration = self.select_next_node(current_node, path, traffic_mode)
            if next_node is None or next_node == current_node:
                # 如果无法找到新的节点或返回到相同的节点，终止搜索
                break
            path.append(next_node)  # 添加节点的索引到路径中
            total_duration += duration  # 累加总时长
            current_node = list(self.graph.nodes())[next_node]  # 更新当前节点
            if current_node == list(self.graph.nodes())[end_index]:
                break  # 如果到达终点，终止循环
        return path, total_duration  # 返回路径（索引列表）

    def select_next_node(self, current_node, path, traffic_mode):
        neighbors = list(self.graph.neighbors(current_node))
        probabilities = []
        durations = []

        current_node_key = str(self.graph.nodes[current_node]['coord'])

        # 收集信息素和时长
        pheromone_list = []
        for neighbor in neighbors:
            neighbor_key = str(self.graph.nodes[neighbor]['coord'])
            pheromone_level = self.pheromone[(current_node_key, neighbor_key)]
            pheromone_list.append(pheromone_level)

            if self.can_turn(current_node_key, neighbor_key, traffic_mode, path):
                edge_length = self.graph.edges[current_node, neighbor]['d1']  # 使用'd1'属性获取边的长度
                travel_time = self.time_recorder.calculate_travel_time(edge_length)
                self.time_recorder.update_relative_time(travel_time)  # 更新相对时间
                wait_time = self.time_recorder.calculate_wait_time(neighbor_key)
                total_duration = travel_time + wait_time
                durations.append(total_duration)
            else:
                durations.append(float('inf'))

        # 信息素归一化和概率计算的部分保持不变
        total_pheromone = sum(pheromone_list)
        normalized_pheromones = [pheromone / total_pheromone for pheromone in pheromone_list]

        for i, neighbor in enumerate(neighbors):
            if durations[i] != float('inf'):
                probability = normalized_pheromones[i] / durations[i] if durations[i] else 0
                probabilities.append(probability)
            else:
                probabilities.append(0)

        # 如果所有概率都为零，则返回None和无穷大
        if all(prob == 0 for prob in probabilities):
            return None, float('inf')

        # 选择下一个节点
        next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
        next_node_key = str(self.graph.nodes[next_node]['coord'])
        return next_node_key, durations[neighbors.index(next_node)]

    def estimate_wait_time(self, current_node, next_node):
        """
        估算在路口的等待时间。
        """
        current_time = ...  # 当前时间
        vehicle_count = self.time_recorder.count_vehicles_in_interval(current_node)
        wait_time = vehicle_count * 0.5  # 每辆车增加0.5分钟的等待时间
        return wait_time

    def run(self, start_index, end_index):
        best_path = None
        best_duration = float('inf')
        first_path_found = None

        for iteration in tqdm(range(self.iterations), desc="Optimizing"):
            iteration_paths = []
            iteration_durations = []

            for ant in range(self.num_ants):
                path, total_duration = self.find_path(start_index, end_index, traffic_mode=1)
                iteration_paths.append(path)
                iteration_durations.append(total_duration)

                if first_path_found is None:
                    first_path_found = {
                        'start': start_index,
                        'end': end_index,
                        'path': path,
                        'duration': total_duration
                    }

                if total_duration < best_duration:
                    best_path = path
                    best_duration = total_duration

            self.update_pheromone(iteration_paths, iteration_durations)

        if first_path_found:
            print(f"First path found: {first_path_found['path']}")
            print(f"Start node index: {first_path_found['start']}, End node index: {first_path_found['end']}")
            print(f"Duration: {first_path_found['duration']} hours")

        if best_path:
            total_travel_time, total_wait_time = self.time_recorder.simulate_path(best_path, self.graph)
            '''total_travel_time = self.time_recorder.get_total_travel_time(best_path, self.graph)
            total_wait_time = self.time_recorder.get_total_wait_time(best_path)'''
            total_duration = total_travel_time + total_wait_time

            print(f"最佳路径: {best_path}")
            print(f"总时长: {total_duration} 小时")
            print(f"总行驶时间: {total_travel_time} 小时")
            print(f"总等待时间: {total_wait_time} 小时")

            # 测试输出：展示每个路口的通行时间记录
        print("路口通行时间记录：")
        for intersection, times in self.time_recorder.intersection_times.items():
            print(f"路口 {intersection}: 通行时间 {times}")

    def update_pheromone(self, paths, durations):
        """
        更新信息素。

        :param paths: 这次迭代中所有蚂蚁的路径。
        :param durations: 对应于每条路径的总时长。
        """
        # 信息素蒸发
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)

        # 信息素增强
        for path, duration in zip(paths, durations):
            for i in range(len(path) - 1):
                if duration != 0:  # 避免除以零
                    # 直接使用路径中的索引
                    index_start = path[i]
                    index_end = path[i + 1]
                    edge = (index_start, index_end)
                    self.pheromone[edge] += 1.0 / duration

    def update_intersection_times(self, path, total_duration):
        """
        更新经过的路口的时间记录。

        :param path: 蚂蚁走过的路径。
        :param total_duration: 从起点到终点的总时长。
        """
        time = 0
        for i, node_index in enumerate(path[:-1]):
            segment_length = self.graph[path[i]][path[i + 1]]['length']
            total_path_length = sum(self.graph[path[j]][path[j + 1]]['length'] for j in range(len(path) - 1))
            segment_duration = total_duration * (segment_length / total_path_length)
            time += segment_duration
            self.time_recorder.record_passing_time(node_index, time)


# 示例代码
if __name__ == "__main__":
    # 载入图形数据
    graph_path = 'data/PAR.graphml'  # 替换为实际的文件路径
    G = read_graphml_with_keys('data/PAR.graphml', ['d0', 'd1'])

    # 创建蚁群优化器实例
    aco = AntColonyOptimizer(G, num_ants=50, evaporation_rate=0.25, iterations=100)

    # 从图中随机选择起点和终点
    start, end = aco.select_start_end_points()

    # 输出选定的起点和终点
    print(f"Selected Start Point: {start}")
    print(f"Selected End Point: {end}")

    # 运行算法
    aco.run(start, end)
