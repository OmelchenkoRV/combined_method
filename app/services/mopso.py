import numpy as np
import copy

from app.models.particle import Particle


class MOPSO:
    def __init__(self, calculation_request, discount_rates=None, num_providers=3,
                 num_particles=50, max_iter=100, bounds=(0, 1)):

        # Extract data from calculation request
        self.price = np.array(calculation_request.price)
        self.support_cost = np.array(calculation_request.support_cost)
        self.daily_orders = np.array(calculation_request.daily_orders)
        self.n_days = calculation_request.n_days

        # Dimensions
        self.k = len(self.price)  # Number of services
        self.m = num_providers  # Number of providers

        # Simulation parameters
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds

        # Create problem parameters based on calculation request

        # Profit coefficients for IT company
        self.d = np.zeros((self.k, self.m))
        for i in range(self.k):
            # Distribute profit potential across providers
            avg_daily_orders = np.mean([orders[i] for orders in self.daily_orders])
            self.d[i, :] = self.price[i] * avg_daily_orders * 2

        # Discount rates
        if discount_rates is None:
            self.r = np.random.rand(self.k, self.m) * 0.1  # Default 0-10% discounts
        else:
            self.r = discount_rates

        # Resource usage coefficients (based on order volume)
        num_resources = 2  # Example: Server and Network resources
        self.beta = np.zeros((self.k, self.m, num_resources))
        for i in range(self.k):
            avg_order_volume = np.mean([orders[i] for orders in self.daily_orders])
            for j in range(self.m):
                # Resource 1: Server usage proportional to order volume
                self.beta[i, j, 0] = avg_order_volume * 0.05
                # Resource 2: Network usage proportional to order volume and price
                self.beta[i, j, 1] = avg_order_volume * self.price[i] * 0.01

        # Resource limits (80% of total resource usage)
        self.T = np.array([np.sum(self.beta[:, :, l]) * 0.8 for l in range(num_resources)])

        # Service costs (from calculation request)
        self.s = np.zeros((self.k, self.m))
        for i in range(self.k):
            # Distribute support costs across providers with some variation
            base_cost = self.support_cost[i]
            for j in range(self.m):
                # Add some variation to support costs between providers
                self.s[i, j] = base_cost * (0.9 + 0.2 * np.random.random())

        # Service prices (from calculation request)
        self.p = np.zeros((self.k, self.m))
        for i in range(self.k):
            for j in range(self.m):
                # Add some variation to prices between providers
                self.p[i, j] = self.price[i] * (0.95 + 0.1 * np.random.random())

        # Price divisor coefficients
        self.b = np.random.rand(self.k, self.m) + 1.0

        # Service dependencies
        num_dep_types = 2  # Technical and business dependencies
        self.G = num_dep_types

        # Dependency coefficients
        self.a_ijg = np.zeros((self.k, self.m, self.G))
        for i in range(self.k):
            for j in range(self.m):
                # Technical dependencies
                self.a_ijg[i, j, 0] = self.price[i] * 0.1
                # Business dependencies
                self.a_ijg[i, j, 1] = np.mean([orders[i] for orders in self.daily_orders]) * 0.2

        # Dependency limits
        self.a_ig = np.zeros((self.k, self.G))
        for i in range(self.k):
            # Technical dependency limits
            self.a_ig[i, 0] = self.price[i] * 0.3
            # Business dependency limits
            self.a_ig[i, 1] = np.max([orders[i] for orders in self.daily_orders]) * 0.5

        # Inter-service dependencies
        self.rho = np.zeros((self.k, self.m, self.k))
        # Create some logical dependencies between services
        for i in range(self.k):
            for j in range(self.m):
                for l in range(self.k):
                    # Set dependencies between related services
                    # For example, services with similar price points may be related
                    if abs(self.price[i] - self.price[l]) < np.mean(self.price) * 0.2:
                        self.rho[i, j, l] = 0.3

        # Initialize particle swarm
        self.particles = [Particle(self.k, self.m, bounds) for _ in range(num_particles)]

        # Initialize archive for non-dominated solutions
        self.archive = []

        # For tracking progress
        self.iter_history = []
        self.best_objectives_history = []

    def evaluate_objectives(self, position):
        """
        Evaluate both objectives: IT company profit and provider profits
        """
        # Calculate average orders for each service
        avg_orders = np.mean(self.daily_orders, axis=0)
        # Convert position to binary-like values for clearer allocation
        binary_position = np.where(position > 0.5, 1.0, 0.0)

        # Calculate IT company profit
        it_profit = 0
        for i in range(self.k):
            for j in range(self.m):
                if binary_position[i, j] > 0:
                    # Base profit from price * orders
                    base_profit = self.price[i] * avg_orders[i]
                    # Apply discount factor
                    discounted_profit = base_profit * (1 - self.r[i, j])
                    # Apply profit coefficient
                    service_profit = discounted_profit * self.d[i, j] / np.sum(self.d)
                    it_profit += service_profit

        # Calculate total provider profit
        provider_profits = 0
        for j in range(self.m):
            for i in range(self.k):
                if binary_position[i, j] > 0:
                    # Revenue = price * (1-discount) * avg_orders
                    revenue = self.p[i, j] * (1 - self.r[i, j]) * avg_orders[i]
                    # Cost = support_cost * avg_orders, with lower cost factor
                    cost = self.s[i, j] * avg_orders[i] * 0.7
                    provider_profits += (revenue - cost)

        # Ensure minimal positive profits
        it_profit = max(1.0, it_profit)
        provider_profits = max(1.0, provider_profits)

        return [it_profit, provider_profits]

    def check_constraints(self, position):
        """
        Check if a position satisfies all constraints
        Returns True if all constraints are satisfied, False otherwise
        """
        # Resource constraints - relaxed to allow more services to be selected
        for l in range(len(self.T)):
            if np.sum(self.beta[:, :, l] * position) > self.T[l] * 5:
                return False

        # Service cost ratio constraints - relaxed
        for i in range(self.k):
            for j in range(self.m):
                if position[i, j] > 0.5 and self.p[i, j] / self.b[i, j] < self.s[i, j] * position[i, j] * 0.5:
                    return False

        # Service dependency constraints - relaxed
        for i in range(self.k):
            for j in range(self.m):
                for g in range(self.G):
                    if position[i, j] > 0.5 and self.a_ijg[i, j, g] * position[i, j] > self.a_ig[i, g] * 3:
                        return False

        # Inter-service dependency constraints - relaxed
        for i in range(self.k):
            for j in range(self.m):
                for l in range(self.m):
                    if j != l and position[i, j] > 0.5 and position[i, l] > 0.5 and self.rho[i, j, l] < 0.05:
                        return False

        return True

    def repair_solution(self, position):
        """
        Repair a solution to make it feasible
        This is a simple repair heuristic that tries to satisfy constraints
        """
        repaired_position = position.copy()

        # Convert to binary-like values (0 or 1) for clarity
        repaired_position = np.where(repaired_position > 0.5, 1.0, 0.0)

        # Ensure at least some services are allocated
        if np.sum(repaired_position) < 1:
            # If no services are selected, select a few of the most profitable ones
            profit_potential = np.zeros((self.k, self.m))
            for i in range(self.k):
                for j in range(self.m):
                    avg_orders = np.mean([orders[i] for orders in self.daily_orders])
                    profit_potential[i, j] = (self.price[i] - self.support_cost[i]) * avg_orders

            # Select top 20% of services by profit potential
            flat_indices = np.argsort(profit_potential.flatten())[-int(self.k * self.m * 0.2):]
            for idx in flat_indices:
                i, j = np.unravel_index(idx, profit_potential.shape)
                repaired_position[i, j] = 1.0

        # Handle other constraints with a simpler approach
        for i in range(self.k):
            for j in range(self.m):
                # If value is close to 1, try to keep it at 1, otherwise set to 0
                if repaired_position[i, j] > 0.5:
                    # Check service cost ratio
                    if self.p[i, j] / self.b[i, j] < self.s[i, j] * 0.1:
                        repaired_position[i, j] = 0
                        continue

                    # Check dependency constraints
                    violation = False
                    for g in range(self.G):
                        if self.a_ijg[i, j, g] * repaired_position[i, j] > self.a_ig[i, g] * 10:
                            violation = True
                            break

                    if violation:
                        repaired_position[i, j] = 0
                else:
                    # If value is small, set to zero for clarity
                    repaired_position[i, j] = 0

        # Handle inter-service dependencies
        for i in range(self.k):
            for j in range(self.m):
                for l in range(self.m):
                    if j != l and repaired_position[i, j] > 0.5 and repaired_position[i, l] > 0.5 and self.rho[
                        i, j, l] < 0.1:
                        # Set one of them to 0 (choose the one with lower profit)
                        profit_j = self.d[i, j] * (1 - self.r[i, j])
                        profit_l = self.d[i, l] * (1 - self.r[i, l])
                        if profit_j < profit_l:
                            repaired_position[i, j] = 0
                        else:
                            repaired_position[i, l] = 0

        return repaired_position

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (Pareto dominance)"""
        better_in_one = False
        for i in range(len(obj1)):
            if obj1[i] < obj2[i]:  # obj1 is worse in one objective
                return False
            if obj1[i] > obj2[i]:  # obj1 is better in at least one objective
                better_in_one = True
        return better_in_one

    def non_dominated_sort(self, particles):
        """
        Perform non-dominated sorting of particles
        Returns list of fronts, where each front is a list of particle indices
        """
        fronts = [[]]
        particle_dominated_by = [[] for _ in range(len(particles))]
        particle_domination_count = [0 for _ in range(len(particles))]

        for i in range(len(particles)):
            for j in range(len(particles)):
                if i != j:
                    if self.dominates(particles[i].current_objectives, particles[j].current_objectives):
                        particle_dominated_by[i].append(j)
                    elif self.dominates(particles[j].current_objectives, particles[i].current_objectives):
                        particle_domination_count[i] += 1

            if particle_domination_count[i] == 0:
                particles[i].rank = 0
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for particle_idx in fronts[i]:
                for dominated_idx in particle_dominated_by[particle_idx]:
                    particle_domination_count[dominated_idx] -= 1
                    if particle_domination_count[dominated_idx] == 0:
                        particles[dominated_idx].rank = i + 1
                        next_front.append(dominated_idx)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove the empty front at the end

    def calculate_crowding_distance(self, particles, front):
        """
        Calculate crowding distance for particles in a front
        """
        if len(front) <= 2:
            for idx in front:
                particles[idx].crowding_distance = float('inf')
            return

        for idx in front:
            particles[idx].crowding_distance = 0

        num_objectives = len(particles[0].current_objectives)

        for obj_idx in range(num_objectives):
            # Sort front by each objective
            front.sort(key=lambda i: particles[i].current_objectives[obj_idx])

            # Extreme points get infinite distance
            particles[front[0]].crowding_distance = float('inf')
            particles[front[-1]].crowding_distance = float('inf')

            # Calculate crowding distance
            obj_range = particles[front[-1]].current_objectives[obj_idx] - particles[front[0]].current_objectives[
                obj_idx]
            if obj_range == 0:
                continue

            for i in range(1, len(front) - 1):
                distance = (particles[front[i + 1]].current_objectives[obj_idx] -
                            particles[front[i - 1]].current_objectives[obj_idx]) / obj_range
                particles[front[i]].crowding_distance += distance

    def select_leader(self):
        """
        Select a leader from the archive using binary tournament selection
        based on crowding distance
        """
        if not self.archive:
            # If archive is empty, return a random particle's position
            return self.particles[np.random.randint(0, len(self.particles))].position

        # Binary tournament selection
        idx1 = np.random.randint(0, len(self.archive))
        idx2 = np.random.randint(0, len(self.archive))

        if self.archive[idx1].rank < self.archive[idx2].rank:
            return self.archive[idx1].position
        elif self.archive[idx1].rank > self.archive[idx2].rank:
            return self.archive[idx2].position
        elif self.archive[idx1].crowding_distance > self.archive[idx2].crowding_distance:
            return self.archive[idx1].position
        else:
            return self.archive[idx2].position

    def update_archive(self):
        """
        Update the archive with non-dominated solutions from the current particles
        """
        # Add all particles to a temporary list
        combined = self.archive + self.particles

        # Perform non-dominated sorting
        fronts = self.non_dominated_sort(combined)

        # Clear current archive
        self.archive = []

        # Add solutions from the first front
        for idx in fronts[0]:
            if idx < len(combined):  # Safety check
                self.archive.append(copy.deepcopy(combined[idx]))

        # Calculate crowding distance for archive
        if fronts[0]:
            self.calculate_crowding_distance(combined, fronts[0])

    def optimize(self):
        """
        Run the MOPSO algorithm

        Returns:
        archive: Final archive of non-dominated solutions
        """
        print("Starting MOPSO optimization...")

        # Initialize particles
        for particle in self.particles:
            # Evaluate initial position
            particle.position = self.repair_solution(particle.position)
            particle.current_objectives = self.evaluate_objectives(particle.position)
            particle.pbest_position = particle.position.copy()
            particle.pbest_objectives = particle.current_objectives.copy()

        # Initialize archive
        self.update_archive()

        # Optimization loop
        for iter_num in range(self.max_iter):
            print(f"Iteration {iter_num + 1}/{self.max_iter}")

            # Update particles
            for particle in self.particles:
                # Select leader
                leader_position = self.select_leader()

                # Update velocity and position with dynamic parameters
                w = 0.5 + np.random.random() * 0.4  # Dynamic inertia weight
                c1 = 1.2 + np.random.random() * 0.6  # Dynamic cognitive coefficient
                c2 = 1.2 + np.random.random() * 0.6  # Dynamic social coefficient

                particle.update_velocity(leader_position, w=w, c1=c1, c2=c2)
                particle.update_position(self.bounds)

                # Repair solution to ensure constraints are satisfied
                particle.position = self.repair_solution(particle.position)

                # Evaluate new position
                particle.current_objectives = self.evaluate_objectives(particle.position)

                # Update personal best
                if self.dominates(particle.current_objectives, particle.pbest_objectives):
                    particle.pbest_position = particle.position.copy()
                    particle.pbest_objectives = particle.current_objectives.copy()
                # If neither dominates, use sum of objectives as tiebreaker
                elif not self.dominates(particle.pbest_objectives, particle.current_objectives):
                    sum_current = sum(particle.current_objectives)
                    sum_pbest = sum(particle.pbest_objectives)
                    if sum_current > sum_pbest:
                        particle.pbest_position = particle.position.copy()
                        particle.pbest_objectives = particle.current_objectives.copy()

            # Update archive
            self.update_archive()

            # Track progress
            self.iter_history.append(iter_num)

            # Find best objectives in current archive
            if self.archive:
                it_profits = [p.current_objectives[0] for p in self.archive]
                provider_profits = [p.current_objectives[1] for p in self.archive]
                max_it_profit = max(it_profits) if it_profits else 0
                max_provider_profit = max(provider_profits) if provider_profits else 0
                self.best_objectives_history.append([max_it_profit, max_provider_profit])

                # Display progress in console
                display_it_profit = max_it_profit
                display_provider_profit = max_provider_profit

                print(f"  Best IT profit: {display_it_profit:.4f}")
                print(f"  Best provider profit: {display_provider_profit:.4f}")
                print(f"  Archive size: {len(self.archive)}")

        print("Optimization complete.")
        print(f"Final archive size: {len(self.archive)}")

        return self.archive

    def get_best_compromise_solution(self):
        """
        Get the best compromise solution from the Pareto front
        using the weighted sum method with equal weights
        """
        if not self.archive:
            return None

        best_score = float('-inf')
        best_solution = None

        for particle in self.archive:
            # Equal weights for both objectives
            score = 0.5 * particle.current_objectives[0] + 0.5 * particle.current_objectives[1]
            if score > best_score:
                best_score = score
                best_solution = particle

        return best_solution

    def optimize_service_selection(self, solution):
        """Optimize the service selection for a given solution"""
        # Calculate profit potential for each service-provider combination
        profit_matrix = np.zeros((self.k, self.m))
        avg_orders = np.mean(self.daily_orders, axis=0)

        for i in range(self.k):
            for j in range(self.m):
                # Calculate IT company profit
                it_profit = self.d[i, j] * (1 - self.r[i, j]) * avg_orders[i]

                # Calculate provider profit
                provider_revenue = self.p[i, j] * (1 - self.r[i, j]) * avg_orders[i]
                provider_cost = self.s[i, j] * avg_orders[i]
                provider_profit = provider_revenue - provider_cost

                # Combined profit potential
                profit_matrix[i, j] = it_profit + provider_profit

        # Create a new position matrix (binary selection matrix)
        position = np.zeros((self.k, self.m))

        # Sort service-provider combinations by profit potential (highest first)
        flat_indices = np.argsort(profit_matrix.flatten())[::-1]

        # Select highest profit combinations while respecting constraints
        selected_services = set()
        selected_providers_count = np.zeros(self.m)
        min_services_to_select = min(max(5, self.k // 4), self.k)

        for idx in flat_indices:
            i, j = np.unravel_index(idx, profit_matrix.shape)

            # Skip if profit is negative or negligible
            if profit_matrix[i, j] <= 0.1:
                continue

            # Set this combination to selected
            position[i, j] = 1.0
            selected_services.add(i)
            selected_providers_count[j] += 1

            # Check constraints after each selection
            if not self.check_constraints(position):
                # If constraints violated, undo this selection
                position[i, j] = 0.0
                selected_services.discard(i)
                selected_providers_count[j] -= 1
                continue

            # If we've selected enough services and all providers have assignments, we can stop
            if (len(selected_services) >= min_services_to_select and
                    np.all(selected_providers_count > 0)):
                break

        # Ensure we have at least some selections even if profitability is low
        if np.sum(position) == 0:
            # Find the least unprofitable combinations
            for idx in flat_indices[:min_services_to_select]:
                i, j = np.unravel_index(idx, profit_matrix.shape)
                position[i, j] = 1.0

        # Update the solution
        solution.position = position
        solution.current_objectives = self.evaluate_objectives(position)

        return solution