# ACO.py
import random
import networkx as nx
from TimeRecorder import TimeRecorder
from IntersectionIndexer import create_intersection_index
from tqdm import tqdm  # 添加tqdm库
import math
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

class AntColonyOptimizer:
    def __init__(self, graph, num_ants, evaporation_rate, iterations):
        self.graph = graph
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromone = {}
        self.intersection_index = create_intersection_index(self.graph)
        self.TimeRecorder = TimeRecorder()
        # 调用select_start_end_points获取起点和终点的节点索引
        start_index, end_index = self.select_start_end_points()

        # 获取起点和终点的坐标，用于初始化信息素
        start_coord = self.graph.nodes[start_index]['coord']
        end_coord = self.graph.nodes[end_index]['coord']
        # 将字符串坐标转换为(x, y)元组格式
        start_coord = tuple(map(float, start_coord.split(',')))
        end_coord = tuple(map(float, end_coord.split(',')))

        # 使用获取到的起点和终点坐标初始化信息素
        self.initialize_pheromone(start_coord, end_coord)

    def parse_coord(self, coord_str):
        """解析坐标字符串，返回(x, y)元组。"""
        x, y = coord_str.split(',')
        return float(x), float(y)

    def distance_to_line(self, point, line_start, line_end):
        """计算点到线段的最短距离。"""
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        proj_length = np.dot(point_vec, line_vec) / np.linalg.norm(line_vec)
        proj = np.array(line_start) + proj_length * line_vec / np.linalg.norm(line_vec)
        return np.linalg.norm(proj - np.array(point))

    def initialize_pheromone(self, start_coord, end_coord):
        """根据起点和终点的连线初始化信息素。"""
        for edge in self.graph.edges:
            node_start, node_end = edge
            start_pos = self.parse_coord(self.graph.nodes[node_start]['coord'])
            end_pos = self.parse_coord(self.graph.nodes[node_end]['coord'])

            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            start_coord = self.parse_coord(self.graph.nodes[self.start_index]['coord'])
            end_coord = self.parse_coord(self.graph.nodes[self.end_index]['coord'])
            distance = self.distance_to_line(mid_point, start_coord, end_coord)
            self.pheromone[edge] = 1 / (1 + distance)

    def calculate_angle(coord1, coord2, coord3):
        """
        计算由三个坐标点定义的两条线段之间的角度。
        参数:
        - coord1, coord2, coord3: 三个坐标点，其中coord2是共同点。
        返回值:
        - angle: 两条线段之间的角度（以360°记角）。
        """
        # 计算向量
        vector1 = (coord2[0] - coord1[0], coord2[1] - coord1[1])
        vector2 = (coord3[0] - coord2[0], coord3[1] - coord2[1])

        # 计算向量的点积和模
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        norm1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        norm2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # 计算角度
        cos_angle = dot_product / (norm1 * norm2)
        angle = math.acos(cos_angle)
        angle_deg = math.degrees(angle)

        # 计算向量叉积以确定角度方向（顺时针或逆时针）
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        if cross_product > 0:
            # 如果叉积为正，说明角度应该在0到180度之间
            return angle_deg
        else:
            # 如果叉积为负，说明角度大于180度，需要调整
            return 360 - angle_deg

    def calculate_cosine_similarity(self, coord1, coord2, coord_target):
        """计算向量(coord1 -> coord2)与(coord1 -> coord_target)之间的余弦相似度。"""
        vector_a = (coord2[0] - coord1[0], coord2[1] - coord1[1])
        vector_b = (coord_target[0] - coord1[0], coord_target[1] - coord1[1])
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a ** 2 for a in vector_a))
        norm_b = math.sqrt(sum(b ** 2 for b in vector_b))
        if norm_a == 0 or norm_b == 0:
            return 0
        cosine_similarity = dot_product / (norm_a * norm_b)
        return cosine_similarity

    # 然后在select_next_node方法中引入对calculate_cosine_similarity的调用来评估方向一致性。



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
        if traffic_mode == 1:  # 自由转弯模式
            return True
        elif traffic_mode == 2:  # 限制转弯模式
            prev_node_index = self.get_previous_node(current_node_index, path)
            if prev_node_index is None:
                return True  # 如果没有上一个节点，说明这是路径的起点，可以自由转弯

            try:
                # 解析节点坐标字符串为数值元组
                prev_coord = self.parse_coord(self.graph.nodes[prev_node_index]['coord'])
                curr_coord = self.parse_coord(self.graph.nodes[current_node_index]['coord'])
                next_coord = self.parse_coord(self.graph.nodes[next_node_index]['coord'])
            except KeyError as e:
                print(f"缺少坐标数据：{e}")
                return False  # 如果任何节点缺少coord属性，则不允许转弯

            # 使用改进的角度计算方法
            angle = self.calculate_angle(prev_coord, curr_coord, next_coord)

            # 允许在0到180度（以360°记角）内的转弯
            if 0 <= angle <= 180:
                return True
            else:
                return False

    def select_start_end_points(self):
        nodes = list(self.graph.nodes())
        start = random.choice(nodes)
        end = random.choice(nodes)
        while end == start:
            end = random.choice(nodes)
        # 直接返回选定的起点和终点节点索引
        return start, end

    def find_path(self, start_index, end_index, traffic_mode):
        """
        对于每只蚂蚁寻找从起点到终点的路径。

        :param start_index: 路径的起点索引。
        :param end_index: 路径的终点索引。
        :param traffic_mode: 交通模式。
        :return: 路径列表和总时长。
        """
        path = [start_index]
        current_node = start_index  # 直接使用节点ID，不需要转换为列表索引

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

        current_node_index = self.intersection_index[current_node]
        current_node_coords = list(self.graph.nodes())[current_node_index]

        # 收集信息素和时长
        pheromone_list = []
        for neighbor in neighbors:
            neighbor_index = self.intersection_index[neighbor]
            # 使用dict.get避免KeyError，假设默认信息素值为0.1
            pheromone_level = self.pheromone.get((current_node_index, neighbor_index), 0.1)
            pheromone_list.append(pheromone_level)

            if self.can_turn(current_node_index, neighbor_index, traffic_mode, path):
                edge_length = self.graph.edges[current_node_coords, neighbor]['length']
                travel_time = self.TimeRecorder.calculate_travel_time(edge_length)
                self.TimeRecorder.update_relative_time(travel_time)  # 更新相对时间
                wait_time = self.TimeRecorder.calculate_wait_time(neighbor_index)
                total_duration = travel_time + wait_time
                durations.append(total_duration)
            else:
                durations.append(float('inf'))

        # 信息素归一化和概率计算
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
        next_node_index = self.intersection_index[next_node]
        return next_node_index, durations[neighbors.index(next_node)]

    def estimate_wait_time(self, current_node, next_node):
        """
        估算在路口的等待时间。
        """
        current_time = ...  # 当前时间
        vehicle_count = self.TimeRecorder.count_vehicles_in_interval(current_node)
        wait_time = vehicle_count * 0.5  # 每辆车增加0.5分钟的等待时间
        return wait_time

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
            self.TimeRecorder.record_passing_time(node_index, time)

    def draw_best_path(self, best_path):
        import matplotlib.pyplot as plt

        coords = [self.parse_coord(self.graph.nodes[node]['d0']) for node in best_path]
        xs, ys = zip(*coords)  # 解包坐标列表

        plt.figure(figsize=(10, 10))
        plt.plot(xs, ys, '-o')  # 绘制路径
        plt.show()

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
            total_travel_time, total_wait_time = self.TimeRecorder.simulate_path(best_path, self.graph)
            '''total_travel_time = self.TimeRecorder.get_total_travel_time(best_path, self.graph)
            total_wait_time = self.TimeRecorder.get_total_wait_time(best_path)'''
            total_duration = total_travel_time + total_wait_time

            print(f"最佳路径: {best_path}")
            print(f"总时长: {total_duration} 小时")
            print(f"总行驶时间: {total_travel_time} 小时")
            print(f"总等待时间: {total_wait_time} 小时")

            # 测试输出：展示每个路口的通行时间记录
        print("路口通行时间记录：")
        for intersection, times in self.TimeRecorder.intersection_times.items():
            print(f"路口 {intersection}: 通行时间 {times}")


# 示例代码
if __name__ == "__main__":
    # 载入图形数据
    graph_path = 'data/PAR.graphml'  # 替换为实际的文件路径
    G = nx.read_graphml(graph_path)

    # 创建蚁群优化器实例
    aco = AntColonyOptimizer(G, num_ants=100, evaporation_rate=0.2, iterations=100)

    # 从图中随机选择起点和终点
    start, end = aco.select_start_end_points()

    # 输出选定的起点和终点
    print(f"Selected Start Point: {start}")
    print(f"Selected End Point: {end}")

    # 运行算法
    aco.run(start, end)