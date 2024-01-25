import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import math


def load_network(path):
    """
    加载街道网络图。
    """
    return nx.read_graphml('data/paris_street_network.graphml')


def calculate_bearing(pointA, pointB):
    """
    计算从点A到点B的方位角（bearing）。
    """
    # 检查是否为Point类型
    if not all(isinstance(point, Point) for point in [pointA, pointB]):
        raise ValueError("Both pointA and pointB must be Shapely Point objects")

    lat1 = math.radians(pointA.y)
    lat2 = math.radians(pointB.y)

    diffLong = math.radians(pointB.x - pointA.x)

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # 将结果转换为0-360度范围内
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def is_straight_or_right_turn(G, u, v):
    if u not in G or v not in G:
        return False

    # 检查 'pos' 属性是否存在，如果不存在则使用节点标识符
    u_pos = G.nodes[u]['pos'] if 'pos' in G.nodes[u] else u
    v_pos = G.nodes[v]['pos'] if 'pos' in G.nodes[v] else v

    # 将字符串坐标转换为元组
    u_pos = tuple(map(float, u_pos.split(',')))
    v_pos = tuple(map(float, v_pos.split(',')))

    u_point = Point(u_pos)
    v_point = Point(v_pos)

    bearing = calculate_bearing(u_point, v_point)

    if 0 <= bearing <= 30 or 330 <= bearing <= 360 or 60 <= bearing <= 120:
        return True

    return False



