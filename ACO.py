# ACO.py
import random
import networkx as nx
from TimeRecorder import TimeRecorder
from IntersectionIndexer import create_intersection_index
from tqdm import tqdm  # 添加tqdm库
import math
import matplotlib.pyplot as plt
import geopandas as gpd



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
        self.intersection_index = create_intersection_index(graph)  # 初始化路口索引
        # 初始化信息素，确保边的每个节点都使用了正确的索引
        self.pheromone = {}

        for edge in self.graph.edges():
            node1_index = self.intersection_index[edge[0]]
            node2_index = self.intersection_index[edge[1]]
            self.pheromone[(node1_index, node2_index)] = 10  # 提高初始信息素水平到 1

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

    def calculate_distance(self, coord1, coord2):
        """计算两点之间的欧氏距离。

        参数:
        - coord1: 第一个点的坐标（x1, y1）。
        - coord2: 第二个点的坐标（x2, y2）。

        返回值:
        - 两点之间的距离。
        """
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

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

    def parse_coord(self, coord_str):
        """将坐标字符串转换为数值元组。

        参数:
        - coord_str: 坐标的字符串表示，格式为"x,y"。

        返回值:
        - 一个包含两个浮点数的元组，表示坐标。
        """
        x_str, y_str = coord_str.split(",")
        return (float(x_str), float(y_str))

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
        return self.intersection_index[start], self.intersection_index[end]

    def find_path(self, start_index, end_index, traffic_mode):
        """
        对于每只蚂蚁寻找从起点到终点的路径。

        :param start_index: 路径的起点索引。
        :param end_index: 路径的终点索引。
        :param traffic_mode: 交通模式。
        :return: 路径列表和总时长。
        """
        path = [start_index]
        current_node = list(self.graph.nodes())[start_index]

        total_duration = 0
        max_iterations = 1000  # 根据需要调整
        found_end = False

        while not found_end and max_iterations > 0:
            next_node, duration = self.select_next_node(current_node, path, traffic_mode,
                                                        list(self.graph.nodes())[end_index])
            if next_node is None or next_node == current_node:
                # 如果无法找到新的节点或返回到相同的节点，尝试其他策略或重启搜索
                break
            path.append(next_node)
            total_duration += duration
            current_node = list(self.graph.nodes())[next_node]
            if current_node == list(self.graph.nodes())[end_index]:
                found_end = True
            max_iterations -= 1

        if not found_end:
            return None, None  # 未找到路径时返回None
        return path, total_duration

    def select_next_node(self, current_node, path, traffic_mode, end_node):
        neighbors = list(self.graph.neighbors(current_node))
        current_coord = self.parse_coord(self.graph.nodes[current_node]['coord'])
        end_node_coord = self.parse_coord(self.graph.nodes[end_node]['coord'])

        alpha = 1  # 信息素重要性因子保持不变
        beta = 5  # 启发式因子重要性，可以进一步调整以强化直线方向的寻路优先级
        direction_factor = 2  # 新增一个因子来强化方向性评估的影响

        probabilities = []
        for neighbor in neighbors:
            neighbor_coord = self.parse_coord(self.graph.nodes[neighbor]['coord'])
            directionality = self.calculate_cosine_similarity(current_coord, neighbor_coord, end_node_coord)
            distance_to_end = self.calculate_distance(neighbor_coord, end_node_coord)

            # 调整启发式值，结合方向性和距离
            heuristic_value = ((1 / distance_to_end) ** beta) * ((directionality + 1) / 2) ** direction_factor
            probabilities.append(heuristic_value)

        # 确保至少有一个合法的方向选择
        if sum(probabilities) == 0:
            return None, float('inf')

        # 根据概率选择下一个节点
        normalized_probabilities = [prob / sum(probabilities) for prob in probabilities]
        next_node_index = random.choices(range(len(neighbors)), weights=normalized_probabilities, k=1)[0]
        next_node = neighbors[next_node_index]

        # 假设边的权重代表旅行时间
        travel_time = self.graph.edges[current_node, next_node]['weight']
        return self.intersection_index[next_node], travel_time

    def estimate_wait_time(self, current_node, next_node):
        """
        估算在路口的等待时间。
        """
        current_time = ...  # 当前时间
        vehicle_count = self.time_recorder.count_vehicles_in_interval(current_node)
        wait_time = vehicle_count * 0.5  # 每辆车增加0.5分钟的等待时间
        return wait_time



    def update_pheromone(self, paths, durations, min_duration):
        """
        更新信息素。

        :param paths: 这次迭代中所有蚂蚁的路径。
        :param durations: 对应于每条路径的总时长。
        """
        # 信息素蒸发
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)

        # 根据路径时长相对于最短路径时长的比例增加信息素
        for path, duration in zip(paths, durations):
            enhancement_factor = min_duration / duration if duration != 0 else 1
            for i in range(len(path) - 1):
                index_start = path[i]
                index_end = path[i + 1]
                edge = (index_start, index_end)
                self.pheromone[edge] += enhancement_factor  # 使用增强因子调整信息素增强量

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
                # 确保total_duration是有效数值
                if total_duration is not None and isinstance(total_duration, (int, float)):
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

                    if path is None or total_duration is None:
                        # 处理未找到路径的情况
                        continue  # 或其他适当的处理方式

            # 迭代结束后找到最短路径的总时长
            min_duration = min(iteration_durations) if iteration_durations else float('inf')
            # 在run方法中更新信息素
            self.update_pheromone(iteration_paths, iteration_durations, min_duration)

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

            '''# 绘制最佳路径
            self.draw_best_path(best_path)'''

        # 测试输出：展示每个路口的通行时间记录
        print("路口通行时间记录：")
        for intersection, times in self.time_recorder.intersection_times.items():
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
