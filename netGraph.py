import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString


def load_shp_and_convert_to_graph(shp_path, length_attr='METERS'):
    gdf = gpd.read_file(shp_path)
    G = nx.DiGraph()

    for _, row in gdf.iterrows():
        if type(row['geometry']) == LineString:
            start, end = row['geometry'].coords[0], row['geometry'].coords[-1]

            # 添加道路属性
            road_attributes = {
                'length': row[length_attr],
                # 可以添加更多属性，如 'speed_limit': row['speed_limit']
            }

            # 添加边到图中，考虑双向道路
            G.add_edge(start, end, **road_attributes)
            # 如果是双向道路，还需要添加反向边
            # G.add_edge(end, start, **road_attributes)

    return G


# 使用函数
shp_path = 'data/streets.shp'  # 替换为你的shp文件路径
graph = load_shp_and_convert_to_graph(shp_path)

# 保存为GraphML格式
nx.write_graphml(graph, 'data/paris_street_network.graphml')  # 替换为你希望保存的GraphML文件路径
