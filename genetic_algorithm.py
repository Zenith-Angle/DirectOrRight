import networkx as nx
import random
import numpy as np
# 假设 free.py 和 DoR.py 在同一目录下，它们包含了路口模拟函数
from free import simulate_free_turns
from DoR import simulate_direct_or_right_turns


# 生成随机点对
def generate_random_points(G, num_pairs=100):
    points = []
    for _ in range(num_pairs):
        start, end = random.sample(list(G.nodes()), 2)
        points.append((start, end))
    return points


# 加载路网图
def load_graph(file_path):
    return nx.read_graphml(file_path)


# 随机生成路口选择规则
def generate_rules(G, mode):
    rules = {}
    for node in G.nodes():
        if mode == 'free':
            rules[node] = random.choice(['直行', '右转', '左转'])
        elif mode == 'restricted':
            rules[node] = random.choice(['直行', '右转'])
    return rules


# 适应度函数
def fitness(rules, G, test_points, mode):
    if mode == 'free':
        return simulate_free_turns(G, rules, test_points)
    elif mode == 'restricted':
        return simulate_direct_or_right_turns(G, rules, test_points)


# 交叉函数
def crossover(parent1, parent2):
    child = {}
    for node in parent1:
        child[node] = parent1[node] if random.random() < 0.5 else parent2[node]
    return child


# 变异函数
def mutate(rules, G, mode):
    node = random.choice(list(G.nodes()))
    if mode == 'free':
        rules[node] = random.choice(['直行', '右转', '左转'])
    elif mode == 'restricted':
        rules[node] = random.choice(['直行', '右转'])
    return rules


# 遗传算法主函数
def genetic_algorithm(G, test_points, mode, generations=100, population_size=100):
    population = [generate_rules(G, mode) for _ in range(population_size)]
    best_fitness = np.inf
    best_solution = None

    for generation in range(generations):
        fitnesses = [fitness(rules, G, test_points, mode) for rules in population]
        # 选择最优解
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_fitness:
            best_fitness = fitnesses[best_idx]
            best_solution = population[best_idx]

        # 选择、交叉和变异
        new_population = []
        while len(new_population) < population_size:
            parents = random.choices(population, weights=np.reciprocal(fitnesses), k=2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, G, mode)
            new_population.append(child)

        population = new_population

    return best_solution


# 测试部分
if __name__ == "__main__":
    G = load_graph(r"data/paris_street_network.graphml")
    test_points = generate_random_points(G, 500)  # 生成500对测试点
    mode = 'free'  # 'free' 或 'restricted'
    best_rules = genetic_algorithm(G, test_points, mode)
    print("最佳路口选择规则:", best_rules)
