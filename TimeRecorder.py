class TimeRecorder:
    def __init__(self):
        self.intersection_times = {}  # 存储每个路口的通行时间记录
        self.relative_time = 0  # 当前相对时间

    def update_relative_time(self, additional_time):
        """
        更新当前的相对时间。

        :param additional_time: 新增的时间（分钟）。
        """
        self.relative_time += additional_time

    def get_current_relative_time(self):
        """
        获取当前的相对时间。

        :return: 当前相对时间（分钟）。
        """
        return self.relative_time

    def record_passing_time(self, intersection):
        """
        记录车辆通过路口的时间（相对于行程开始的时间）。
        :param intersection: 路口编号。
        """
        if intersection not in self.intersection_times:
            self.intersection_times[intersection] = []
        self.intersection_times[intersection].append(self.relative_time)

    def count_vehicles_in_interval(self, intersection):
        """
        计算在当前相对时间点前后一定时间内通过路口的车辆数量。
        :param intersection: 路口编号。
        :return: 数量。
        """
        if intersection not in self.intersection_times:
            return 0

        time_interval = 0.02  # 您可以根据需要调整这个时间间隔
        time_list = self.intersection_times[intersection]
        count = sum(
            1 for t in time_list if self.relative_time - time_interval <= t <= self.relative_time + time_interval)
        return count

    def calculate_travel_time(self, edge_length):
        """
        计算给定距离的行驶时间。
        :param edge_length: 边的长度（米）。
        :return: 行驶时间（小时）。
        """
        edge_length = float(edge_length)

        speed = 40000.0 / 60 # 假设速度（米/小时），注意单位转换
        return edge_length / speed

    def calculate_wait_time(self, intersection):
        """
        计算路口的等待时间。
        :param intersection: 当前节点。
        :return: 等待时间（小时）。
        """
        vehicle_count = self.count_vehicles_in_interval(intersection)
        return vehicle_count * 0.01  # 每辆车增加0.01小时的等待时间


    def get_total_travel_time(self, path, graph):
        total_travel_time = 0
        for i in range(len(path) - 1):
            try:
                edge_length = graph[path[i]][path[i + 1]]['length']
                travel_time = self.calculate_travel_time(edge_length)
                total_travel_time += travel_time
            except KeyError:
                continue  # 如果找不到边，则跳过
        return total_travel_time

    def get_total_wait_time(self, path):
        total_wait_time = 0
        for node_index in path:
            if node_index in self.intersection_times:
                wait_time = self.intersection_times[node_index][-1] * 0.5
                total_wait_time += wait_time
        return total_wait_time

    def simulate_path(self, path, graph):
        total_travel_time = 0
        total_wait_time = 0

        for i in range(len(path) - 1):
            start_node_id = path[i]
            end_node_id = path[i + 1]
            edge_length = graph.edges[start_node_id, end_node_id]['length']
            travel_time = self.calculate_travel_time(edge_length)
            total_travel_time += travel_time
            self.update_relative_time(travel_time)
            wait_time = self.calculate_wait_time(end_node_id)
            total_wait_time += wait_time

        return total_travel_time, total_wait_time

    def reset_relative_time(self):
        """
        重置相对时间。
        """
        self.relative_time = 0