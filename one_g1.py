import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 引入3D工具
from deap import base, creator, tools

# 創建最小化問題
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 初始化基因演算法的工具箱
toolbox = base.Toolbox()

# 定義基因個體，這裡每個個體有 3 個變數，表示 3 維的 Sphere function
toolbox.register("attr_float", random.uniform, -100, 100)  # 初始化範圍
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 10) #三個基因
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Sphere function 作為適應度函數
def evalSphere(individual):
    return sum(x**2 for x in individual),

# 設置基因演算法的交叉、變異、選擇操作
toolbox.register("evaluate", evalSphere)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # 高斯變異
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament Selection

# 主程式
def main():
    # 初始化種群
    pop = toolbox.population(n=100)
    
    # 評估初始種群的適應度
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # 設置交叉和變異的機率
    CXPB, MUTPB = 0.5, 0.2
    
    # 記錄當前的最優個體
    best_ind = tools.selBest(pop, 1)[0]
    
    # 用於記錄每一代的最佳適應度
    best_fitnesses = []
    best_individuals = []  # 用於記錄每一代的最佳個體

    # 進化過程
    print("開始進化...")
    no_improvement_count = 0  # 用於記錄無改善的世代數
    max_no_improvement = 50  # 允許無改善的最大世代數

    best_fitness = best_ind.fitness.values[0]  # 初始化最佳適應度

    for gen in range(200000):
        # 選擇下一代的個體
        offspring = toolbox.select(pop, len(pop))  # Tournament Selection
        offspring = list(map(toolbox.clone, offspring))
        
        # 應用交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # 應用變異操作
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 評估新個體
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # 更新種群
        pop[:] = offspring

        # 找到當前最優解
        current_best_ind = tools.selBest(pop, 1)[0]
        if current_best_ind.fitness.values[0] < best_ind.fitness.values[0]:
            best_ind = current_best_ind
            best_fitness = best_ind.fitness.values[0]
            no_improvement_count = 0  # 重置無改善計數
        else:
            no_improvement_count += 1  # 增加無改善計數
        
        # 記錄當前世代的最佳適應度和最佳個體
        best_fitnesses.append(best_ind.fitness.values[0])
        best_individuals.append(best_ind)
        
        # 每一代輸出當前的最佳結果
        print(f"世代 {gen}, 最佳適應度: {best_ind.fitness.values[0]}, 最佳個體: {best_ind}")   

        # 如果連續無改善的世代數達到最大值，則退出
        if no_improvement_count >= max_no_improvement:
            print("無改善達到最大次數，提前退出!")
            break

    # 繪製最佳適應度隨世代變化的圖
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitnesses, label='Best Fitness')
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend()
    plt.show()

    # 繪製 3D 曲面圖
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 創建曲面的網格數據
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Sphere function

    # 繪製曲面
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

    # 繪製每個個體的位置
    pop_x = [ind[0] for ind in pop]
    pop_y = [ind[1] for ind in pop]
    pop_z = [x**2 + y**2 for x, y in zip(pop_x, pop_y)]  # 根據 X 和 Y 計算 Z 值

    # 只繪製點，而不連接線
    ax.scatter(pop_x, pop_y, pop_z, color='b', label='Population Points', alpha=0.6)

    # 繪製最佳個體的點
    best_x = [ind[0] for ind in best_individuals]
    best_y = [ind[1] for ind in best_individuals]
    best_z = [x**2 + y**2 for x, y in zip(best_x, best_y)]  # 根據最佳個體的 X 和 Y 計算 Z 值
    ax.scatter(best_x, best_y, best_z, color='r', s=100, label='Best Individuals', alpha=0.8)
     # 最終結果
    print("\n進化完成。")
    print(f"最優個體: {best_ind}")
    best_fitness = format(best_ind.fitness.values[0], '.30f')
    print(f"最優適應度: {best_fitness}")
    ax.set_title('3D Surface Plot of Sphere Function with Population Points')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    plt.show()

  

if __name__ == "__main__":
    main()
