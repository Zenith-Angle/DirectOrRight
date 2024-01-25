# DoR.py - 模拟只能直行或右转的通行方式
from rule import traffic_window


def simulate_direct_or_right_turns(intersection, incoming_traffic):
    # 过滤掉不符合直行或右转规则的车辆
    filtered_traffic = [vehicle for vehicle in incoming_traffic if vehicle['turn'] in ['straight', 'right']]
    return traffic_window(intersection, filtered_traffic)


def main():
    # 示例：创建模拟环境
    intersection = {'directions': ['N', 'S', 'E', 'W']}
    incoming_traffic = [{'direction': 'N', 'arrival_time': 0, 'turn': 'right'},
                        {'direction': 'E', 'arrival_time': 1, 'turn': 'straight'},
                        {'direction': 'S', 'arrival_time': 2, 'turn': 'left'},
                        {'direction': 'W', 'arrival_time': 3, 'turn': 'right'}]

    # 调用模拟函数
    traffic_order = simulate_direct_or_right_turns(intersection, incoming_traffic)
    print("通过路口的车辆顺序:", traffic_order)


if __name__ == "__main__":
    main()
