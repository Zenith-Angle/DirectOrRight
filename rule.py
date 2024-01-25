# rule.py - 定义通行规则

from collections import deque
from itertools import cycle


def traffic_window(intersection, incoming_traffic, window_duration=1):
    """
    实现通行窗口规则。
    :param intersection: 路口对象，包含各个方向。
    :param incoming_traffic: 即将到达路口的车辆列表。
    :param window_duration: 通行窗口时长（分钟）。
    :return: 返回通过路口的车辆顺序。
    """
    # 初始化车辆队列
    traffic_queues = {direction: deque() for direction in intersection['directions']}

    # 按到达时间将车辆加入相应队列
    for vehicle in incoming_traffic:
        traffic_queues[vehicle['direction']].append(vehicle)

    traffic_order = []
    current_time = 0

    # 循环遍历每个方向
    for direction in cycle(intersection['directions']):
        if all(not queue for queue in traffic_queues.values()):
            break  # 所有队列都空时结束

        while traffic_queues[direction] and traffic_queues[direction][0]['arrival_time'] <= current_time:
            # 将车辆从队列中移出并加入到通行顺序列表中
            traffic_order.append(traffic_queues[direction].popleft())

        current_time += window_duration  # 更新当前时间

    return traffic_order


# 示例用法
if __name__ == "__main__":
    intersection = {'directions': ['N', 'S', 'E', 'W']}
    incoming_traffic = [{'direction': 'N', 'arrival_time': 0},
                        {'direction': 'E', 'arrival_time': 1},
                        {'direction': 'S', 'arrival_time': 2},
                        {'direction': 'W', 'arrival_time': 3},
                        {'direction': 'N', 'arrival_time': 4}]

    traffic_order = traffic_window(intersection, incoming_traffic)
    print("通过路口的车辆顺序:", [v['direction'] for v in traffic_order])
