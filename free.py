# free.py - 模拟可任意转向的通行方式
from rule import traffic_window


def simulate_free_turns(intersection, incoming_traffic):
    # 所有方向的车辆都被允许通过
    return traffic_window(intersection, incoming_traffic)


def main():
    # 示例：创建模拟环境
    intersection = {'directions': ['N', 'S', 'E', 'W']}
    incoming_traffic = [{'direction': 'N', 'arrival_time': 0},
                        {'direction': 'E', 'arrival_time': 1},
                        {'direction': 'S', 'arrival_time': 2},
                        {'direction': 'W', 'arrival_time': 3}]

    # 调用模拟函数
    traffic_order = simulate_free_turns(intersection, incoming_traffic)
    print("通过路口的车辆顺序:", traffic_order)


if __name__ == "__main__":
    main()
