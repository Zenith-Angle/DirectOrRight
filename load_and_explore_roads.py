# 导入必要的库
import geopandas as gpd


def load_and_explore_shp(file_path):
    # 加载shp文件
    road_network = gpd.read_file(file_path)

    # 显示数据的前几行
    print("数据的前五行：")
    print(road_network.head())

    # 显示数据列的名称和类型
    print("\n数据列的名称和类型：")
    print(road_network.dtypes)

    # 检查是否有缺失值
    print("\n检查缺失值：")
    print(road_network.isnull().sum())

    return road_network


# 设置文件路径
file_path = r"D:\Direct_or_Right\直行向右\data\streets.shp"

# 调用函数并加载数据
road_network = load_and_explore_shp(file_path)
