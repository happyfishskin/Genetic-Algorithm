import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools

# 創建最小化問題
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class GeneticAlgorithmOptimizer:
    def __init__(self, population_size=100, cxpb=0.5, mutpb=0.2, ngen=200000):
        self.population_size = population_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen

        # 初始化 DEAP 工具箱
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -500, 500)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, 30)  # 10維
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.eval_schwefel)

        # 使用不同的交叉策略
        self.toolbox.register("mate", tools.cxOnePoint)  # 单点交叉
        self.toolbox.register("mutate", tools.mutUniformInt, low=-500, up=500, indpb=0.2)  # Uniform mutation
        self.toolbox.register("select", tools.selRoulette)  # Roulette Wheel Selection

    def eval_schwefel(self, individual):
        return 418.9829 * len(individual) - sum(x * np.sin(np.sqrt(abs(x))) for x in individual),

    def initialize_population(self):
        return self.toolbox.population(n=self.population_size)

    def evaluate_population(self, population):
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

    def evolve(self, population):
        best_individual = tools.selBest(population, 1)[0]
        global_best_fitness = best_individual.fitness.values[0]
        
        no_improvement_count = 0
        max_no_improvement = 10
        best_fitness_history = []

        for gen in range(self.ngen):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉操作
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 变异操作
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 更新适应度
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.evaluate_population(invalid_ind)

            # 用新的后代替换原始种群
            population[:] = offspring
            
            # 保留最佳个体
            best_individual = tools.selBest(population, 1)[0]
            best_fitness = best_individual.fitness.values[0]
            best_fitness_history.append(best_fitness)

            # 更新全局最佳适应度
            if best_fitness < global_best_fitness:
                global_best_fitness = best_fitness

            print(f"世代 {gen}, 最佳适应度: {best_fitness}, 最佳个体: {best_individual}")

            # 如果最佳适应度没有改善
            if len(best_fitness_history) > 1 and abs(best_fitness_history[-1] - best_fitness_history[-2]) < 1e-6:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= max_no_improvement:
                print("无改善达到最大次数，提前退出!")
                break
                
        print(f"全局最佳适应度: {global_best_fitness}")
        return best_individual

    def run(self):
        population = self.initialize_population()
        self.evaluate_population(population)
        best_individual = self.evolve(population)
        return best_individual, population

    def plot(self, population, best_individual, filename="3d_plot.png"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.plot_schwefel_surface(ax)
        self.plot_population(ax, population, best_individual)

        ax.set_box_aspect([1, 1, 0.2])
        plt.legend()
        plt.show()

    def plot_schwefel_surface(self, ax):
        # 繪製 3D Schwefel 函數的表面
        u = np.linspace(-500, 500, 100)
        x, y = np.meshgrid(u, u)
        z = 418.9829 * 2 - (x * np.sin(np.sqrt(abs(x))) + y * np.sin(np.sqrt(abs(y))))
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.6)
        ax.set_zlim(-2000, 2000)  # 根據 Schwefel 函數的範圍調整

    def plot_population(self, ax, population, best_individual):
        x_vals = [ind[0] for ind in population]
        y_vals = [ind[1] for ind in population]
        z_vals = [self.eval_schwefel([x, y])[0] for x, y in zip(x_vals, y_vals)]

        ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o', s=50, label='Population')

        # 繪製最佳個體
        best_x = best_individual[0]
        best_y = best_individual[1]
        best_z = self.eval_schwefel([best_x, best_y])[0]
        ax.scatter(best_x, best_y, best_z, c='b', marker='^', s=100, label='Best Individual')

# 主程式
if __name__ == "__main__":
    optimizer = GeneticAlgorithmOptimizer(population_size=100, cxpb=0.5, mutpb=0.2, ngen=100)
    best_individual, population = optimizer.run()
    print(f"最終最佳個體: {best_individual}")
    optimizer.plot(population, best_individual, filename="ga_3d_plot.png")
