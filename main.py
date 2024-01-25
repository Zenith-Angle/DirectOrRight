from deap import base, tools
import network_utils
import genetic_algorithm
import visualization
import random


def main():
    # 加载网络图
    G = network_utils.load_network('data/paris_street_network.graphml')
    print("节点数量:", len(G.nodes()))
    print("边数量:", len(G.edges()))

    # 初始化遗传算法工具箱
    toolbox = base.Toolbox()
    toolbox.register("individual", genetic_algorithm.init_individual, G, 10)  # 路径长度为10，可根据需要调整
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", genetic_algorithm.evaluate, G)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 创建初始种群
    population = toolbox.population(n=100)  # 种群数量可调整

    # 运行遗传算法
    NGEN = 50  # 代数
    for gen in range(NGEN):
        # 选择和克隆
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    # 获取最佳路径
    best_ind = tools.selBest(population, 1)[0]

    # 可视化最佳路径
    visualization.visualize_path(G, best_ind)

    return best_ind




if __name__ == "__main__":
    best_path = main()
    print("Best path found:", best_path)
