# ACO.py
import math
import random
import networkx as nx
from TimeRecorder import TimeRecorder
from IntersectionIndexer import create_intersection_index
from tqdm import tqdm  # 添加tqdm库

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
            self.pheromone[(node1_index, node2_index)] = 10  # 提高初始信息素水平到 10

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

    def can_turn(self, current_node, next_node, traffic_mode, path):
        """
        根据交通规则判断是否可以从当前节点转向下一个节点。

        :param current_node: 当前节点。
        :param next_node: 下一个节点。
        :param traffic_mode: 交通模式，1代表自由转弯，2代表限制转弯。
        :param path: 蚂蚁走过的路径。
        :return: 是否可以转向。
        """
        if traffic_mode == 1:  # 自由转弯模式
            return True
        elif traffic_mode == 2:  # 限制转弯模式
            prev_node = self.get_previous_node(current_node, path)
            if prev_node is None:
                return True  # 如果没有上一个节点，说明这是路径的起点，可以自由转弯

            # 获取节点的坐标
            prev_coord = self.graph.nodes[prev_node]['coord']
            curr_coord = self.graph.nodes[current_node]['coord']
            next_coord = self.graph.nodes[next_node]['coord']

            # 计算夹角
            angle = self.calculate_angle(prev_coord, curr_coord, next_coord)

            # 判断是否是直行或右转
            return 160 <= angle <= 180 or 80 <= angle <= 100  # 这里的角度范围可以根据实际情况调整

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

        current_node_index = self.intersection_index[current_node]
        current_node_coords = list(self.graph.nodes())[current_node_index]

        # 收集信息素和时长
        pheromone_list = []
        for neighbor in neighbors:
            neighbor_index = self.intersection_index[neighbor]
            pheromone_level = self.pheromone[(current_node_index, neighbor_index)]
            pheromone_list.append(pheromone_level)

            if self.can_turn(current_node_coords, neighbor, traffic_mode, path):
                edge_length = self.graph.edges[current_node_coords, neighbor]['length']
                travel_time = self.time_recorder.calculate_travel_time(edge_length)
                self.time_recorder.update_relative_time(travel_time)  # 更新相对时间
                wait_time = self.time_recorder.calculate_wait_time(neighbor_index)
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
        next_node_index = self.intersection_index[next_node]
        return next_node_index, durations[neighbors.index(next_node)]

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
    graph_path = 'data/paris_street_network.graphml'  # 替换为实际的文件路径
    G = nx.read_graphml(graph_path)

    # 创建蚁群优化器实例
    aco = AntColonyOptimizer(G, num_ants=10, evaporation_rate=0.5, iterations=50)

    # 从图中随机选择起点和终点
    start, end = aco.select_start_end_points()

    # 输出选定的起点和终点
    print(f"Selected Start Point: {start}")
    print(f"Selected End Point: {end}")

    # 运行算法
    aco.run(start, end)