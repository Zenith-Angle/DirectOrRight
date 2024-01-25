import networkx as nx
import random
from DoR import simulate_direct_or_right_turns
from free import simulate_free_turns


def load_road_network(graphml_path):
    """加载并返回GraphML格式的路网图"""
    return nx.read_graphml(graphml_path)


def apply_traffic_rule(road_network, start, end, rule_function):
    """根据交通规则调整网络，并计算最短路径"""
    adjusted_network = road_network.copy()
    for u, v in road_network.edges():
        if not rule_function({'direction': u, 'arrival_time': 0}, {'direction': v, 'arrival_time': 0}):
            adjusted_network.remove_edge(u, v)
    return nx.shortest_path(adjusted_network, source=start, target=end, weight='length')


def generate_random_point_pairs(road_network, num_pairs):
    """随机生成起点和终点对"""
    nodes = list(road_network.nodes)
    return [random.sample(nodes, 2) for _ in range(num_pairs)]


def main():
    # 加载路网图
    road_network = load_road_network('data/paris_street_network.graphml')

    # 生成随机点对
    point_pairs = generate_random_point_pairs(road_network, 50)

    # 对每对点计算两种规则下的最短路径
    for start_point, end_point in point_pairs:
        shortest_path_dor = apply_traffic_rule(road_network, start_point, end_point, simulate_direct_or_right_turns)
        shortest_path_free = apply_traffic_rule(road_network, start_point, end_point, simulate_free_turns)

        print("起点:", start_point, "终点:", end_point)
        print("直行或右转规则下的最短路径:", shortest_path_dor)
        print("任意转向规则下的最短路径:", shortest_path_free)


if __name__ == "__main__":
    main()
