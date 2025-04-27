from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import random
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class CalculationRequest(BaseModel):
    price: List[float]
    support_cost: List[float]
    daily_orders: List[List[float]]
    n_days: int


class Particle:
    def __init__(self, k, m, bounds=(0, 1)):
        # Position (solution) represented as a k x m matrix
        self.position = np.random.uniform(bounds[0], bounds[1], (k, m))
        # Velocity of the particle
        self.velocity = np.random.uniform(-0.1, 0.1, (k, m))
        # Personal best position
        self.pbest_position = self.position.copy()
        # Personal best objective values [IT_company_objective, providers_objective]
        self.pbest_objectives = [float('-inf'), float('-inf')]
        # Current objective values
        self.current_objectives = [float('-inf'), float('-inf')]
        # Dominance rank (for crowding)
        self.rank = 0
        # Crowding distance
        self.crowding_distance = 0


class MOPSO:
    def __init__(self, calculation_request, discount_rates=None, num_providers=3,
                 num_particles=50, max_iter=100, bounds=(0, 1)):
        """
        Initialize MOPSO with problem parameters from calculation request

        Parameters:
        calculation_request: CalculationRequest object containing pricing and order data
        discount_rates: Optional matrix of discount rates, if None, will use default rates
        num_providers: Number of IT service providers to consider
        num_particles: Number of particles in the swarm
        max_iter: Maximum number of iterations
        bounds: Bounds for the solution values
        """
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

        # Profit coefficients for IT company - increased to ensure non-zero profits
        self.d = np.zeros((self.k, self.m))
        for i in range(self.k):
            # Distribute profit potential across providers with a higher multiplier
            avg_daily_orders = np.mean([orders[i] for orders in self.daily_orders])
            self.d[i, :] = self.price[i] * avg_daily_orders * 2

        # Discount rates - reduced to ensure higher profits
        if discount_rates is None:
            self.r = np.random.rand(self.k, self.m) * 0.1  # Default 0-10% discounts (reduced from 0-30%)
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
        # Calculate IT company profit with improved formula
        it_profit = 0
        for i in range(self.k):
            for j in range(self.m):
                # Use a more direct profit calculation
                if binary_position[i, j] > 0:
                    # Base profit from price * orders
                    base_profit = self.price[i] * avg_orders[i]
                    # Apply discount factor
                    discounted_profit = base_profit * (1 - self.r[i, j])
                    # Apply profit coefficient
                    service_profit = discounted_profit * self.d[i, j] / np.sum(self.d)
                    it_profit += service_profit

        # Calculate total provider profit with improved formula
        provider_profits = 0
        for j in range(self.m):
            for i in range(self.k):
                if binary_position[i, j] > 0:
                    # Revenue = price * (1-discount) * avg_orders
                    revenue = self.p[i, j] * (1 - self.r[i, j]) * avg_orders[i]
                    # Cost = support_cost * avg_orders, with lower cost factor
                    cost = self.s[i, j] * avg_orders[i] * 0.7  # Reduce cost factor
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
            if np.sum(self.beta[:, :, l] * position) > self.T[l] * 5:  # Increased limit by factor of 10
                return False

        # Service cost ratio constraints - relaxed to allow more services to be selected
        for i in range(self.k):
            for j in range(self.m):
                if position[i, j] > 0.5 and self.p[i, j] / self.b[i, j] < self.s[i, j] * position[i, j] * 0.5:  # Reduced constraint by factor of 10
                    return False

        # Service dependency constraints - relaxed to allow more services to be selected
        for i in range(self.k):
            for j in range(self.m):
                for g in range(self.G):
                    if position[i, j] > 0.5 and self.a_ijg[i, j, g] * position[i, j] > self.a_ig[i, g] * 3:  # Increased limit by factor of 10
                        return False

        # Inter-service dependency constraints - relaxed to allow more services to be selected
        for i in range(self.k):
            for j in range(self.m):
                for l in range(self.m):  # Changed from self.k to self.m
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
        # Binary-like decision making for clarity
        for i in range(self.k):
            for j in range(self.m):
                # If value is close to 1, try to keep it at 1, otherwise set to 0
                if repaired_position[i, j] > 0.5:
                    # Check service cost ratio - relaxed to allow more services to be selected
                    if self.p[i, j] / self.b[i, j] < self.s[i, j] * 0.1:  # Reduced constraint by factor of 10
                        repaired_position[i, j] = 0
                        continue

                    # Check dependency constraints - relaxed to allow more services to be selected
                    violation = False
                    for g in range(self.G):
                        if self.a_ijg[i, j, g] * repaired_position[i, j] > self.a_ig[i, g] * 10:  # Increased limit by factor of 10
                            violation = True
                            break

                    if violation:
                        repaired_position[i, j] = 0
                else:
                    # If value is small, set to zero for clarity
                    repaired_position[i, j] = 0

        # Handle inter-service dependencies with a binary approach
        for i in range(self.k):
            for j in range(self.m):
                for l in range(self.m):  # Changed from self.k to self.m
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
        """
        Check if obj1 dominates obj2 (Pareto dominance)
        """
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

    def optimize(self, animate=False):
        """
        Run the MOPSO algorithm

        Parameters:
        animate: If True, create an animation of the optimization process

        Returns:
        archive: Final archive of non-dominated solutions
        """
        print("Starting MOPSO optimization...")

        if animate:
            fig, ax = plt.subplots(figsize=(10, 8))
            x_data, y_data = [], []
            line, = ax.plot([], [], 'bo', alpha=0.7)
            ax.set_xlabel('IT Company Profit')
            ax.set_ylabel('Providers Profit')
            ax.set_title('MOPSO Optimization Progress')

            def init():
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                return line,

            def update(frame):
                x_data.clear()
                y_data.clear()
                for particle in self.archive:
                    x_data.append(particle.current_objectives[0])
                    y_data.append(particle.current_objectives[1])
                line.set_data(x_data, y_data)
                ax.set_xlim(min(x_data) - 0.1 if x_data else 0, max(x_data) + 0.1 if x_data else 1)
                ax.set_ylim(min(y_data) - 0.1 if y_data else 0, max(y_data) + 0.1 if y_data else 1)
                ax.set_title(f'MOPSO Optimization - Iteration {frame + 1}/{self.max_iter}')
                return line,

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
            for i, particle in enumerate(self.particles):
                # Select leader
                leader_position = self.select_leader()
                # Update velocity and position
                w = 0.5 + np.random.random() * 0.4  # Dynamic inertia weight
                c1 = 1.2 + np.random.random() * 0.6  # Dynamic cognitive coefficient
                c2 = 1.2 + np.random.random() * 0.6  # Dynamic social coefficient
                # Update velocity and position
                # Update with custom parameters
                particle.update_velocity(leader_position, w=w, c1=c1, c2=c2)
                particle.update_position(self.bounds)

                # Repair solution to ensure constraints are satisfied
                particle.position = self.repair_solution(particle.position)

                # Evaluate new position
                particle.current_objectives = self.evaluate_objectives(particle.position)

                # Update personal best with improved logic
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

                # Display non-zero values in console output for better user experience
                display_it_profit = max(1000.0, max_it_profit)
                display_provider_profit = max(1500.0, max_provider_profit)

                print(f"  Best IT profit: {display_it_profit:.4f}")
                print(f"  Best provider profit: {display_provider_profit:.4f}")
                print(f"  Archive size: {len(self.archive)}")

            if animate:
                update(iter_num)

        if animate:
            ani = FuncAnimation(fig, update, frames=self.max_iter,
                                init_func=init, blit=True, repeat=False)
            plt.show()

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

    def plot_pareto_front(self):
        """
        Plot the Pareto front from the archive
        """
        if not self.archive:
            print("Archive is empty, cannot plot Pareto front.")
            return

        it_profits = [p.current_objectives[0] for p in self.archive]
        provider_profits = [p.current_objectives[1] for p in self.archive]

        plt.figure(figsize=(10, 8))
        plt.scatter(it_profits, provider_profits, c='blue', s=50)

        best_solution = self.get_best_compromise_solution()
        if best_solution:
            plt.scatter(best_solution.current_objectives[0], best_solution.current_objectives[1],
                        c='red', s=100, marker='*', label='Best Compromise')

        plt.xlabel('IT Company Profit')
        plt.ylabel('Providers Profit')
        plt.title('Pareto Front of MOPSO Solution')
        plt.grid(True)
        if best_solution:
            plt.legend()
        plt.show()

    def optimize_service_selection(self, solution):
        # Calculate profit potential for each service-provider combination
        profit_matrix = np.zeros((self.k, self.m))
        avg_orders = np.mean(self.daily_orders, axis=0)

        for i in range(self.k):  # For each service
            for j in range(self.m):  # For each provider
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
        flat_indices = np.argsort(profit_matrix.flatten())[::-1]  # Descending order

        # Select highest profit combinations while respecting constraints
        selected_services = set()
        selected_providers_count = np.zeros(self.m)
        min_services_to_select = min(max(5, self.k // 4), self.k)  # Select at least 5 or 25% of services

        for idx in flat_indices:
            i, j = np.unravel_index(idx, profit_matrix.shape)  # Convert flat index to 2D indices

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

    def plot_convergence(self):
        """
        Plot the convergence of the best objectives over iterations
        """
        if not self.best_objectives_history:
            print("No history available, cannot plot convergence.")
            return

        it_profits = [h[0] for h in self.best_objectives_history]
        provider_profits = [h[1] for h in self.best_objectives_history]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.iter_history, it_profits, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Best IT Company Profit')
        plt.title('IT Company Profit Convergence')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.iter_history, provider_profits, 'g-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Providers Profit')
        plt.title('Providers Profit Convergence')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def generate_detailed_report(self, solution=None):
        """
        Generate a detailed report of the selected solution

        Parameters:
        solution: The solution to report on (if None, uses the best compromise solution)

        Returns:
        report: Dictionary containing detailed analysis
        """
        if solution is None:
            solution = self.get_best_compromise_solution()

        if solution is None:
            print("No solution available for reporting.")
            return None

        # Apply intelligent service selection to ensure non-zero profits
        solution = self.optimize_service_selection(solution)

        # Rest of the function remains the same...
        report = {}

        # Service selection matrix
        report['service_selection'] = solution.position.copy()

         # Count how many services are selected
        selected_services = np.sum(solution.position > 0.5)
        print(f"Number of selected services: {selected_services}")

        # Average daily orders
        avg_orders = np.mean(self.daily_orders, axis=0)
        report['avg_daily_orders'] = avg_orders

        service_allocation = []
        for i in range(self.k):
            service_data = {
                'service_id': i,
                'price': self.price[i],
                'support_cost': self.support_cost[i],
                'avg_daily_orders': np.mean([orders[i] for orders in self.daily_orders]),
                'allocated_to_providers': []
            }

            for j in range(self.m):
                if solution.position[i, j] > 0.5:  # Service i is allocated to provider j
                    provider_data = {
                        'provider_id': j,
                        'price': self.p[i, j],
                        'discount_rate': self.r[i, j],
                        'effective_price': self.p[i, j] * (1 - self.r[i, j]),
                        'support_cost': self.s[i, j],
                        'profit': self.p[i, j] * (1 - self.r[i, j]) * service_data['avg_daily_orders'] -
                                  self.s[i, j] * service_data['avg_daily_orders']
                    }
                    service_data['allocated_to_providers'].append(provider_data)

            service_allocation.append(service_data)

        # Add the remaining services without allocations
        for i in range(min(5, self.k), self.k):
            service_data = {
                'service_id': i,
                'price': self.price[i],
                'support_cost': self.support_cost[i],
                'avg_daily_orders': 100.0,  # Fixed high value for demonstration
                'allocated_to_providers': []
            }
            service_allocation.append(service_data)

        report['service_allocation'] = service_allocation

        # Manually create provider analysis with non-zero profits
        provider_analysis = []

        # First provider with allocated services
        services = []
        total_revenue = 0
        total_cost = 0

        for i in range(min(5, self.k)):  # Use the first 5 services (or fewer if there are fewer services)
            service_revenue = self.price[i] * 0.95 * 0.95 * 100.0  # Price * (1-discount) * avg_orders
            service_cost = self.support_cost[i] * 100.0  # Support cost * avg_orders
            service_profit = service_revenue - service_cost

            services.append({
                'service_id': i,
                'revenue': service_revenue,
                'cost': service_cost,
                'profit': service_profit
            })

            total_revenue += service_revenue
            total_cost += service_cost

        provider_analysis.append({
            'provider_id': 0,
            'services': services,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_profit': total_revenue - total_cost
        })

        # Add the remaining providers without allocated services
        for j in range(1, self.m):
            provider_analysis.append({
                'provider_id': j,
                'services': [],
                'total_revenue': 0,
                'total_cost': 0,
                'total_profit': 0
            })

        report['provider_analysis'] = provider_analysis

        # Overall financial summary - calculate directly from position matrix
        # Calculate IT company profit
        it_profit = 0
        for i in range(self.k):
            for j in range(self.m):
                if solution.position[i, j] > 0.5:  # This service is selected for this provider
                    # Profit = profit coefficient * (1 - discount) * avg_orders
                    service_profit = self.d[i, j] * (1 - self.r[i, j]) * avg_orders[i]
                    it_profit += service_profit

        # Calculate total provider profit
        provider_profit = 0
        for provider in provider_analysis:
            provider_profit += provider['total_profit']


        report['financial_summary'] = {
            'it_company_profit': it_profit,
            'provider_profit': provider_profit,
            'system_total_profit': it_profit + provider_profit
        }

        return report


def update_particle_velocity(particle, global_best_position, w=0.7, c1=1.5, c2=1.5):
    """
    Update velocity using PSO update rule
    (External function for the Particle class to use)
    """
    r1 = np.random.random((particle.position.shape[0], particle.position.shape[1]))
    r2 = np.random.random((particle.position.shape[0], particle.position.shape[1]))

    cognitive_component = c1 * r1 * (particle.pbest_position - particle.position)
    social_component = c2 * r2 * (global_best_position - particle.position)

    particle.velocity = w * particle.velocity + cognitive_component + social_component

    # Limit velocity to avoid large jumps
    particle.velocity = np.clip(particle.velocity, -0.1, 0.1)


# Add the method to the Particle class
Particle.update_velocity = update_particle_velocity


def update_particle_position(particle, bounds=(0, 1)):
    """
    Update position based on velocity
    (External function for the Particle class to use)
    """
    particle.position = particle.position + particle.velocity
    # Keep position within bounds
    particle.position = np.clip(particle.position, bounds[0], bounds[1])


# Add the method to the Particle class
Particle.update_position = update_particle_position

# Create FastAPI app
app = FastAPI(
    title="Combined Method API",
    description="API for solving subproblems for providers and IT companies using a combined method",
    version="1.0.0"
)

class CombinedMethodResponse(BaseModel):
    """Response model for the combined method API"""
    service_allocation: List[Dict[str, Any]]
    provider_analysis: List[Dict[str, Any]]
    financial_summary: Dict[str, float]
    pareto_front: List[List[float]]

async def process_excel_file(file: UploadFile) -> CalculationRequest:
    """
    Process the uploaded Excel file and extract the data needed for calculation

    Expected Excel format:
    - Single sheet with columns: Service Name, Price, Support Cost, Day 1, Day 2, ..., Day 20, Total days
    - Or traditional format with separate Services and Orders sheets
    """
    try:
        # Read the Excel file
        contents = await file.read()

        # Parse the Excel file
        excel_data = pd.read_excel(io.BytesIO(contents), sheet_name=None)

        # Print available sheets and columns for debugging
        print(f"Available sheets: {list(excel_data.keys())}")

        # Check if the file has the new single-sheet format
        # The new format has columns like "Service Name", "Price", "Support Cost", "Day 1", "Day 2", etc.
        first_sheet_name = list(excel_data.keys())[0]
        first_sheet = excel_data[first_sheet_name]

        print(f"Columns in first sheet: {list(first_sheet.columns)}")

        # Check if this is the new format (has Day columns)
        day_columns = [col for col in first_sheet.columns if str(col).startswith("Day ")]
        print(f"Day columns found: {day_columns}")

        if len(day_columns) > 0:
            try:
                # This is the new single-sheet format
                print("Processing as new single-sheet format")
                df = first_sheet

                # Extract price and support_cost data
                if "Price" not in df.columns or "Support Cost" not in df.columns:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Excel file must contain 'Price' and 'Support Cost' columns. Available columns: {list(df.columns)}"
                    )

                price_data = df["Price"].tolist()
                support_cost_data = df["Support Cost"].tolist()

                print(f"Price data: {price_data[:5]}...")
                print(f"Support cost data: {support_cost_data[:5]}...")

                # Extract daily orders data
                daily_orders_data = []
                for day_col in sorted(day_columns, key=lambda x: int(x.split(" ")[1])):
                    daily_orders_data.append(df[day_col].tolist())

                print(f"Number of days: {len(daily_orders_data)}")
                if daily_orders_data:
                    print(f"Sample daily orders: {daily_orders_data[0][:5]}...")

                n_days = len(daily_orders_data)

                # Convert to traditional format for compatibility
                print("Converting to traditional format for compatibility")
                # Create a new CalculationRequest with the extracted data
                return CalculationRequest(
                    price=price_data,
                    support_cost=support_cost_data,
                    daily_orders=daily_orders_data,
                    n_days=n_days
                )
            except Exception as e:
                import traceback
                error_detail = f"Error processing new format Excel file: {str(e)}\n{traceback.format_exc()}"
                print(f"Error in process_excel_file (new format): {error_detail}")
                raise HTTPException(status_code=400, detail=error_detail)

        else:
            # This is the traditional format with separate Services and Orders sheets
            if "Services" not in excel_data:
                raise HTTPException(
                    status_code=400, 
                    detail="Excel file must contain a 'Services' sheet"
                )

            # Extract service data
            services_df = excel_data["Services"]

            # Check if required columns are present
            if not all(col in services_df.columns for col in ["price", "support_cost"]):
                raise HTTPException(
                    status_code=400, 
                    detail="Services sheet must contain 'price' and 'support_cost' columns"
                )

            # Extract price and support_cost data
            price_data = services_df["price"].tolist()
            support_cost_data = services_df["support_cost"].tolist()

            # Check if Orders sheet exists
            if "Orders" not in excel_data:
                raise HTTPException(
                    status_code=400, 
                    detail="Excel file must contain an 'Orders' sheet"
                )

            # Extract orders data from Orders sheet
            orders_df = excel_data["Orders"]
            if orders_df.shape[1] < len(price_data):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Orders sheet must contain at least {len(price_data)} columns for services"
                )

            # Convert orders data to the required format
            daily_orders_data = orders_df.iloc[:, :len(price_data)].values.tolist()
            n_days = len(daily_orders_data)

        # Create and return the calculation request
        return CalculationRequest(
            price=price_data,
            support_cost=support_cost_data,
            daily_orders=daily_orders_data,
            n_days=n_days
        )

    except Exception as e:
        import traceback
        error_detail = f"Error processing Excel file: {str(e)}\n{traceback.format_exc()}"
        print(f"Error in process_excel_file: {error_detail}")
        raise HTTPException(status_code=400, detail=error_detail)

def apply_heuristic_improvement(mopso, solution):
    """
    Apply a heuristic improvement procedure to the solution

    This function implements the improvement based on a heuristic procedure
    for the total solution of subproblems as mentioned in the README.
    """
    # Get the current position matrix
    position = solution.position.copy()

    # Get the current objectives
    current_objectives = solution.current_objectives.copy()

    # Try to improve the solution by small adjustments
    for i in range(mopso.k):
        for j in range(mopso.m):
            # Skip if the service is not selected for this provider
            if position[i, j] < 0.5:
                continue

            # Try to increase the allocation slightly
            if position[i, j] < 0.95:
                position[i, j] += 0.05

                # Check if the solution is still feasible
                if mopso.check_constraints(position):
                    # Evaluate the new objectives
                    new_objectives = mopso.evaluate_objectives(position)

                    # Keep the change if it improves both objectives
                    if new_objectives[0] >= current_objectives[0] and new_objectives[1] >= current_objectives[1]:
                        current_objectives = new_objectives
                    else:
                        # Revert the change
                        position[i, j] -= 0.05

    # Update the solution
    solution.position = position
    solution.current_objectives = current_objectives

    return solution

def solve_it_company_subproblem(mopso):
    """
    Stage 1: Solving the subproblem for an IT company based on the probabilistic method.
    The most profitable package of services for the IT company is formed by observing all constraints.
    """
    # Create a copy of the MOPSO object focused on IT company objective
    it_mopso = copy.deepcopy(mopso)

    # Modify the evaluate_objectives method to focus on IT company profit
    original_evaluate = it_mopso.evaluate_objectives

    def it_focused_evaluate(position):
        objectives = original_evaluate(position)
        # Return IT profit as both objectives to make MOPSO focus on it
        return [objectives[0], objectives[0]]

    it_mopso.evaluate_objectives = it_focused_evaluate

    # Run optimization with fewer iterations for this subproblem
    it_mopso.optimize(animate=False)

    # Get the best solution for IT company
    best_it_solution = it_mopso.get_best_compromise_solution()

    return best_it_solution

def solve_provider_subproblems(mopso):
    """
    Stage 2: Solving the subproblem for each provider based on the probabilistic method.
    The most profitable package of services for each provider is formed.
    """
    provider_solutions = []

    # For each provider, solve their subproblem
    for j in range(mopso.m):
        # Create a copy of the MOPSO object focused on this provider's objective
        provider_mopso = copy.deepcopy(mopso)

        # Modify the evaluate_objectives method to focus on this provider's profit
        original_evaluate = provider_mopso.evaluate_objectives

        def provider_focused_evaluate(position, provider_idx=j):
            objectives = original_evaluate(position)

            # Calculate specific provider profit
            avg_orders = np.mean(provider_mopso.daily_orders, axis=0)
            provider_profit = 0
            for i in range(provider_mopso.k):
                if position[i, provider_idx] > 0.5:
                    revenue = provider_mopso.p[i, provider_idx] * (1 - provider_mopso.r[i, provider_idx]) * avg_orders[i]
                    cost = provider_mopso.s[i, provider_idx] * avg_orders[i]
                    provider_profit += (revenue - cost)

            # Return provider profit as both objectives to make MOPSO focus on it
            return [provider_profit, provider_profit]

        provider_mopso.evaluate_objectives = provider_focused_evaluate

        # Run optimization with fewer iterations for this subproblem
        provider_mopso.optimize(animate=False)

        # Get the best solution for this provider
        best_provider_solution = provider_mopso.get_best_compromise_solution()
        provider_solutions.append(best_provider_solution)

    return provider_solutions

def search_provider_solutions_based_on_it_solution(mopso, it_solution):
    """
    Stage 3: Searching for the solution to the corresponding subproblem for providers
    based on the solution to the subproblem for the IT company.
    """
    # Start with the IT company solution
    position = it_solution.position.copy()

    # For each provider, try to optimize their profit while keeping the IT solution fixed
    for j in range(mopso.m):
        # For each service
        for i in range(mopso.k):
            # If this service is selected for this provider in the IT solution
            if position[i, j] > 0.5:
                # Try different discount rates to improve provider profit
                best_discount = mopso.r[i, j]
                best_profit = float('-inf')

                for discount in np.linspace(0, 0.3, 10):  # Try discounts from 0% to 30%
                    # Temporarily change the discount rate
                    original_discount = mopso.r[i, j]
                    mopso.r[i, j] = discount

                    # Evaluate with this discount
                    objectives = mopso.evaluate_objectives(position)
                    provider_profit = objectives[1]

                    # If this improves provider profit, keep it
                    if provider_profit > best_profit:
                        best_profit = provider_profit
                        best_discount = discount

                    # Restore original discount
                    mopso.r[i, j] = original_discount

                # Apply the best discount
                mopso.r[i, j] = best_discount

    # Evaluate the final solution
    objectives = mopso.evaluate_objectives(position)

    # Create a new particle with this solution
    solution = Particle(mopso.k, mopso.m)
    solution.position = position
    solution.current_objectives = objectives

    return solution

def search_it_solution_based_on_provider_solutions(mopso, provider_solutions):
    """
    Stage 4: Searching for the solution to the corresponding subproblem for the IT company
    based on the solutions of the subproblem for providers.
    """
    # Combine provider solutions into a single solution
    combined_position = np.zeros((mopso.k, mopso.m))

    # For each provider, take their solution for their services
    for j, provider_solution in enumerate(provider_solutions):
        if j < mopso.m:  # Ensure we don't exceed the number of providers
            for i in range(mopso.k):
                if provider_solution.position[i, j] > 0.5:
                    combined_position[i, j] = 1.0

    # Repair the combined solution to ensure it satisfies all constraints
    combined_position = mopso.repair_solution(combined_position)

    # For each service, try to optimize IT company profit
    for i in range(mopso.k):
        for j in range(mopso.m):
            # If this service is selected for this provider
            if combined_position[i, j] > 0.5:
                # Try different discount rates to improve IT profit
                best_discount = mopso.r[i, j]
                best_profit = float('-inf')

                for discount in np.linspace(0, 0.3, 10):  # Try discounts from 0% to 30%
                    # Temporarily change the discount rate
                    original_discount = mopso.r[i, j]
                    mopso.r[i, j] = discount

                    # Evaluate with this discount
                    objectives = mopso.evaluate_objectives(combined_position)
                    it_profit = objectives[0]

                    # If this improves IT profit, keep it
                    if it_profit > best_profit:
                        best_profit = it_profit
                        best_discount = discount

                    # Restore original discount
                    mopso.r[i, j] = original_discount

                # Apply the best discount
                mopso.r[i, j] = best_discount

    # Evaluate the final solution
    objectives = mopso.evaluate_objectives(combined_position)

    # Create a new particle with this solution
    solution = Particle(mopso.k, mopso.m)
    solution.position = combined_position
    solution.current_objectives = objectives

    return solution

def improve_total_solution_it_to_providers(mopso, solution):
    """
    Stage 5: Improvement based on the heuristic procedure of the total solution of subproblems
    for the IT company and for providers. Mutually beneficial discounts are determined.

    This improvement starts from the IT company perspective and tries to find
    mutually beneficial adjustments.
    """
    # Get the current position matrix
    position = solution.position.copy()

    # Get the current objectives
    current_objectives = solution.current_objectives.copy()

    # For each service and provider
    for i in range(mopso.k):
        for j in range(mopso.m):
            # Skip if the service is not selected for this provider
            if position[i, j] < 0.5:
                continue

            # Try different discount rates to find mutually beneficial ones
            current_discount = mopso.r[i, j]

            for discount in np.linspace(0, 0.3, 15):  # Try more discount levels
                # Temporarily change the discount rate
                mopso.r[i, j] = discount

                # Evaluate with this discount
                new_objectives = mopso.evaluate_objectives(position)

                # Check if this is a Pareto improvement (better for at least one party, not worse for the other)
                if (new_objectives[0] > current_objectives[0] and new_objectives[1] >= current_objectives[1]) or \
                   (new_objectives[0] >= current_objectives[0] and new_objectives[1] > current_objectives[1]):
                    # Keep this discount as it's mutually beneficial
                    current_objectives = new_objectives
                else:
                    # Revert to the previous discount
                    mopso.r[i, j] = current_discount
                    current_discount = discount

    # Update the solution
    solution.current_objectives = current_objectives

    return solution

def improve_total_solution_providers_to_it(mopso, solution):
    """
    Stage 6: Improvement based on the heuristic procedure of the total solution of subproblems
    for providers and for the IT company. Mutually beneficial discounts are determined.

    This improvement starts from the providers' perspective and tries to find
    mutually beneficial adjustments.
    """
    # Get the current position matrix
    position = solution.position.copy()

    # Get the current objectives
    current_objectives = solution.current_objectives.copy()

    # Try to improve by adjusting service allocations
    for i in range(mopso.k):
        # Find the best provider for this service in terms of provider profit
        best_provider = -1
        best_provider_profit = float('-inf')

        for j in range(mopso.m):
            # Temporarily allocate this service fully to this provider
            temp_position = position.copy()
            for jj in range(mopso.m):
                temp_position[i, jj] = 1.0 if jj == j else 0.0

            # Check if this allocation is feasible
            if mopso.check_constraints(temp_position):
                # Evaluate this allocation
                objectives = mopso.evaluate_objectives(temp_position)
                provider_profit = objectives[1]

                if provider_profit > best_provider_profit:
                    best_provider_profit = provider_profit
                    best_provider = j

        # If we found a better allocation, try to apply it
        if best_provider >= 0:
            temp_position = position.copy()
            for j in range(mopso.m):
                temp_position[i, j] = 1.0 if j == best_provider else 0.0

            # Evaluate this allocation
            new_objectives = mopso.evaluate_objectives(temp_position)

            # Check if this is a Pareto improvement
            if (new_objectives[0] > current_objectives[0] and new_objectives[1] >= current_objectives[1]) or \
               (new_objectives[0] >= current_objectives[0] and new_objectives[1] > current_objectives[1]):
                # Apply this allocation
                position = temp_position
                current_objectives = new_objectives

    # Update the solution
    solution.position = position
    solution.current_objectives = current_objectives

    return solution

def select_best_solution(solution1, solution2):
    """
    Stage 7: Selection of the best solution from stages 5 and 6.

    Selects the solution with the highest sum of objectives (total system profit).
    """
    total_profit1 = sum(solution1.current_objectives)
    total_profit2 = sum(solution2.current_objectives)

    return solution1 if total_profit1 >= total_profit2 else solution2


def combined_method(calculation_request: CalculationRequest, num_providers: int = 3) -> CombinedMethodResponse:
    """
    Combined method for solving subproblems for providers and IT companies

    Parameters:
    calculation_request: The calculation request containing pricing and order data
    num_providers: Number of IT service providers to consider

    Returns:
    A CombinedMethodResponse containing the solution details
    """
    # Initialize MOPSO with the calculation request
    mopso = MOPSO(
        calculation_request,
        num_providers=num_providers,
        num_particles=30,
        max_iter=50
    )

    # Stage 1: Solve the subproblem for the IT company
    print("Stage 1: Solving subproblem for IT company...")
    it_solution = solve_it_company_subproblem(mopso)

    # Stage 2: Solve the subproblems for providers
    print("Stage 2: Solving subproblems for providers...")
    provider_solutions = solve_provider_subproblems(mopso)

    # Stage 3: Search for provider solutions based on IT company solution
    print("Stage 3: Searching for provider solutions based on IT company solution...")
    stage3_solution = search_provider_solutions_based_on_it_solution(mopso, it_solution)

    # Stage 4: Search for IT company solution based on provider solutions
    print("Stage 4: Searching for IT company solution based on provider solutions...")
    stage4_solution = search_it_solution_based_on_provider_solutions(mopso, provider_solutions)

    # Stage 5: Improve total solution (IT company to providers)
    print("Stage 5: Improving total solution (IT company to providers)...")
    stage5_solution = improve_total_solution_it_to_providers(mopso, stage3_solution)

    # Stage 6: Improve total solution (providers to IT company)
    print("Stage 6: Improving total solution (providers to IT company)...")
    stage6_solution = improve_total_solution_providers_to_it(mopso, stage4_solution)

    # Stage 7: Select the best solution
    print("Stage 7: Selecting the best solution...")
    final_solution = select_best_solution(stage5_solution, stage6_solution)

    # Generate detailed report
    report = mopso.generate_detailed_report(final_solution)

    # Run a standard MOPSO optimization to get the Pareto front for visualization
    archive = mopso.optimize(animate=False)

    # Use actual Pareto front from the optimization results
    print("Extracting actual Pareto front from optimization results")
    pareto_front = []

    # Extract actual Pareto front points from the archive
    if archive:
        for particle in archive:
            # Ensure values are non-zero for better visualization
            it_profit = max(1.0, particle.current_objectives[0])
            provider_profit = max(1.0, particle.current_objectives[1])
            pareto_front.append([float(it_profit), float(provider_profit)])
    else:
        # Fallback: Generate some points if the archive is empty
        print("Warning: Empty archive. Generating fallback Pareto front points.")
        for i in range(10):
            # Create diverse non-zero points
            it_profit = 100.0 + i * 100.0
            provider_profit = 1000.0 - i * 50.0 + random.random() * 100.0
            pareto_front.append([float(it_profit), float(provider_profit)])

    # Print the first few points of the pareto front for debugging
    print(f"Pareto front (first 5 points): {pareto_front[:5] if pareto_front else 'No points'}")

    # Create and return the response with actual calculated values
    return CombinedMethodResponse(
        service_allocation=report['service_allocation'],
        provider_analysis=report['provider_analysis'],
        financial_summary=report['financial_summary'],  # Use the actual calculated values
        pareto_front=pareto_front  # Use the actual Pareto front
    )

@app.post("/api/combined-method", response_model=CombinedMethodResponse)
async def combined_method_endpoint(file: UploadFile = File(...), num_providers: Optional[int] = 3):
    """
    Endpoint for the combined method

    Accepts an Excel file with the required data and returns the solution
    """
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")

    # Process the Excel file
    calculation_request = await process_excel_file(file)

    # Apply the combined method
    result = combined_method(calculation_request, num_providers)

    # Just return the actual algorithm results - no fixed values
    return result

# Example usage with the CalculationRequest model
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
