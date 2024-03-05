import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString


def create_network_from_shp(filepath):
    # 读取Shapefile
    gdf = gpd.read_file(filepath)

    # 创建空的Graph对象
    G = nx.DiGraph()

    node_id = 0
    node_map = {}

    # 遍历GeoDataFrame中的每一行
    for _, row in gdf.iterrows():
        geometries = row.geometry.geoms if isinstance(row.geometry, MultiLineString) else [row.geometry]

        for geom in geometries:
            start_point = geom.coords[0]
            end_point = geom.coords[-1]

            if start_point not in node_map:
                node_map[start_point] = node_id
                G.add_node(node_id, coord=start_point)
                start_node_id = node_id
                node_id += 1
            else:
                start_node_id = node_map[start_point]

            if end_point not in node_map:
                node_map[end_point] = node_id
                G.add_node(node_id, coord=end_point)
                end_node_id = node_id
                node_id += 1
            else:
                end_node_id = node_map[end_point]

            distance = geom.length
            G.add_edge(start_node_id, end_node_id, weight=distance)

    return G

# 替换为你的Shapefile路径
filepath = 'data/streets.shp'
G = create_network_from_shp(filepath)

# 演示如何访问节点坐标
for node in G.nodes(data=True):
    print(f"Node {node[0]}: Coord = {node[1]['coord']}")

# 演示如何访问边的权重
for edge in G.edges(data=True):
    print(f"Edge from {edge[0]} to {edge[1]}: Weight = {edge[2]['weight']}")
