import numpy as np
class Apso:
    def __init__(self, lb, ub, w_max, w_min, c1, c2):
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.maxv = (ub - lb) * 0.2
        self.minv = -self.maxv

    def forward(self, func, swarmsize, maxiter, tol, patience):
        # 初始化粒子位置和速度
        particles = np.random.uniform(self.lb, self.ub, (swarmsize, self.dim))
        velocities = np.random.uniform(self.minv, self.maxv, (swarmsize, self.dim))

        # 初始化个体最佳位置和全局最佳位置
        pbest = particles.copy()
        pbest_fitness = np.array([func(p) for p in pbest])

        gbest = pbest[np.argmax(pbest_fitness)]
        gbest_fitness = max(pbest_fitness)

        best_gbest_history = [gbest]
        best_fitness_history = [gbest_fitness]
        no_improvement_count = 0

        for iteration in range(maxiter):
            w = self.w_max - (self.w_max - self.w_min) * iteration / maxiter
            phi1 = self.c1 * np.random.rand(swarmsize, self.dim)
            phi2 = self.c2 * np.random.rand(swarmsize, self.dim)

            velocities = w * velocities + phi1 * (pbest - particles) + phi2 * (gbest - particles)
            velocities = np.clip(velocities, self.minv, self.maxv)
            particles += velocities
            particles = np.clip(particles, self.lb, self.ub)
            current_fitness = np.array([func(p) for p in particles])
            update_indices = current_fitness < pbest_fitness
            pbest[update_indices] = particles[update_indices]
            pbest_fitness[update_indices] = current_fitness[update_indices]
            if max(current_fitness) > gbest_fitness:
                gbest = particles[np.argmax(current_fitness)]
                gbest_fitness = max(current_fitness)
                best_gbest_history.append(gbest)
                best_fitness_history.append(gbest_fitness)

            if len(best_fitness_history) > patience and np.std(best_fitness_history[-patience:]) < tol:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping at iteration {iteration + 1}")
                    break
            else:
                no_improvement_count = 0
            gbest_fitness = max(best_fitness_history)
            print(f"Iteration {iteration + 1}/{maxiter}: Maximum daily oil production of the well group = {gbest_fitness}")

        gbest_fitness = max(best_fitness_history)
        max_index = best_fitness_history.index(gbest_fitness)
        gbest = best_gbest_history[max_index]
        return gbest, gbest_fitness

