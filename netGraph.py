import geopandas as gpd
import networkx as nx
from shapely.geometry import Point


def load_shp_to_networkx(shp_path):
    # 加载Shapefile
    gdf = gpd.read_file(shp_path)

    # 创建空的networkx图
    G = nx.Graph()

    # 为了简化示例，假设每条线的起点和终点代表图中的边
    for index, row in gdf.iterrows():
        # 获取线的起点和终点坐标
        start_point = row.geometry.coords[0]
        end_point = row.geometry.coords[-1]

        # 添加节点。如果已存在，则不会重复添加
        G.add_node(start_point, coord=start_point)
        G.add_node(end_point, coord=end_point)

        # 添加边
        G.add_edge(start_point, end_point)

    return G


# 调用函数，替换'shp_path'为实际的Shapefile路径
shp_path = 'streets.shp'
G = load_shp_to_networkx(shp_path)

# 打印图信息，验证导入是否成功
print(f"图中节点数: {G.number_of_nodes()}")
print(f"图中边数: {G.number_of_edges()}")
