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

        # Profit coefficients for IT company
        self.d = np.zeros((self.k, self.m))
        for i in range(self.k):
            # Distribute profit potential across providers
            avg_daily_orders = np.mean([orders[i] for orders in self.daily_orders])
            self.d[i, :] = self.price[i] * avg_daily_orders / self.m

        # Discount rates
        if discount_rates is None:
            self.r = np.random.rand(self.k, self.m) * 0.3  # Default 0-30% discounts
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

        # Calculate IT company profit
        it_profit = 0
        for i in range(self.k):
            for j in range(self.m):
                # Profit = (price - discount) * selection * avg_orders
                service_profit = self.d[i, j] * (1 - self.r[i, j]) * position[i, j]
                it_profit += service_profit

        # Calculate total provider profit
        provider_profits = 0
        for j in range(self.m):
            for i in range(self.k):
                if position[i, j] > 0.5:  # This service is selected for this provider
                    # Revenue = price * (1-discount) * avg_orders
                    revenue = self.p[i, j] * (1 - self.r[i, j]) * avg_orders[i]
                    # Cost = support_cost * avg_orders
                    cost = self.s[i, j] * avg_orders[i]
                    provider_profits += (revenue - cost)

        return [it_profit, provider_profits]

    def check_constraints(self, position):
        """
        Check if a position satisfies all constraints
        Returns True if all constraints are satisfied, False otherwise
        """
        # Resource constraints
        for l in range(len(self.T)):
            if np.sum(self.beta[:, :, l] * position) > self.T[l]:
                return False

        # Service cost ratio constraints
        for i in range(self.k):
            for j in range(self.m):
                if position[i, j] > 0 and self.p[i, j] / self.b[i, j] < self.s[i, j] * position[i, j]:
                    return False

        # Service dependency constraints
        for i in range(self.k):
            for j in range(self.m):
                for g in range(self.G):
                    if position[i, j] > 0 and self.a_ijg[i, j, g] * position[i, j] > self.a_ig[i, g]:
                        return False

        # Inter-service dependency constraints (relaxed for binary decision variables)
        for i in range(self.k):
            for j in range(self.m):
                for l in range(self.k):
                    if position[i, j] > 0.5 and position[i, l] > 0.5 and self.rho[i, j, l] < 0.5:
                        return False

        return True

    def repair_solution(self, position):
        """
        Repair a solution to make it feasible
        This is a simple repair heuristic that tries to satisfy constraints
        """
        repaired_position = position.copy()

        # First, handle resource constraints
        for l in range(len(self.T)):
            while np.sum(self.beta[:, :, l] * repaired_position) > self.T[l]:
                # Find the service with highest resource usage
                flat_index = np.argmax(self.beta[:, :, l] * repaired_position)
                i, j = np.unravel_index(flat_index, (self.k, self.m))
                # Reduce its selection
                repaired_position[i, j] *= 0.9

        # Handle other constraints with a simpler approach
        # Binary-like decision making for clarity
        for i in range(self.k):
            for j in range(self.m):
                # If value is close to 1, try to keep it at 1, otherwise set to 0
                if repaired_position[i, j] > 0.5:
                    # Check service cost ratio
                    if self.p[i, j] / self.b[i, j] < self.s[i, j]:
                        repaired_position[i, j] = 0
                        continue

                    # Check dependency constraints
                    violation = False
                    for g in range(self.G):
                        if self.a_ijg[i, j, g] * repaired_position[i, j] > self.a_ig[i, g]:
                            violation = True
                            break

                    if violation:
                        repaired_position[i, j] = 0
                else:
                    # If value is small, set to zero for clarity
                    repaired_position[i, j] = 0

        # Handle inter-service dependencies with a binary approach
        for i in range(self.k):
            for l in range(self.k):
                for j in range(self.m):
                    if repaired_position[i, j] > 0.5 and repaired_position[i, l] > 0.5 and self.rho[i, j, l] < 0.5:
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
                particle.update_velocity(leader_position)
                particle.update_position(self.bounds)

                # Repair solution to ensure constraints are satisfied
                particle.position = self.repair_solution(particle.position)

                # Evaluate new position
                particle.current_objectives = self.evaluate_objectives(particle.position)

                # Update personal best if current position dominates personal best
                if self.dominates(particle.current_objectives, particle.pbest_objectives):
                    particle.pbest_position = particle.position.copy()
                    particle.pbest_objectives = particle.current_objectives.copy()
                # If neither dominates, randomly decide
                elif not self.dominates(particle.pbest_objectives, particle.current_objectives):
                    if random.random() < 0.5:
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

                print(f"  Best IT profit: {max_it_profit:.4f}")
                print(f"  Best provider profit: {max_provider_profit:.4f}")
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

        report = {}

        # Service selection matrix
        report['service_selection'] = solution.position.copy()

        # Average daily orders
        avg_orders = np.mean(self.daily_orders, axis=0)
        report['avg_daily_orders'] = avg_orders

        # Per-service allocation
        service_allocation = []
        for i in range(self.k):
            service_data = {
                'service_id': i,
                'price': self.price[i],
                'support_cost': self.support_cost[i],
                'avg_daily_orders': avg_orders[i],
                'allocated_to_providers': []
            }

            for j in range(self.m):
                if solution.position[i, j] > 0.5:
                    provider_data = {
                        'provider_id': j,
                        'price': self.p[i, j],
                        'discount_rate': self.r[i, j],
                        'effective_price': self.p[i, j] * (1 - self.r[i, j]),
                        'support_cost': self.s[i, j],
                        'profit': self.p[i, j] * (1 - self.r[i, j]) * avg_orders[i] - self.s[i, j] * avg_orders[i]
                    }
                    service_data['allocated_to_providers'].append(provider_data)

            service_allocation.append(service_data)

        report['service_allocation'] = service_allocation

        # Provider analysis
        provider_analysis = []
        for j in range(self.m):
            services = []
            total_revenue = 0
            total_cost = 0

            for i in range(self.k):
                if solution.position[i, j] > 0.5:
                    service_revenue = self.p[i, j] * (1 - self.r[i, j]) * avg_orders[i]
                    service_cost = self.s[i, j] * avg_orders[i]
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
                'provider_id': j,
                'services': services,
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'total_profit': total_revenue - total_cost
            })

        report['provider_analysis'] = provider_analysis

        # Overall financial summary
        report['financial_summary'] = {
            'it_company_profit': solution.current_objectives[0],
            'total_provider_profit': solution.current_objectives[1],
            'system_total_profit': sum(solution.current_objectives)
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
    - Sheet "Services": columns for service_id, price, support_cost
    - Sheet "Orders": daily orders for each service (rows=days, columns=services)
    """
    try:
        # Read the Excel file
        contents = await file.read()

        # Parse the Excel file
        excel_data = pd.read_excel(io.BytesIO(contents), sheet_name=None)

        if "Services" not in excel_data or "Orders" not in excel_data:
            raise HTTPException(
                status_code=400, 
                detail="Excel file must contain 'Services' and 'Orders' sheets"
            )

        # Extract service data
        services_df = excel_data["Services"]
        if not all(col in services_df.columns for col in ["service_id", "price", "support_cost"]):
            raise HTTPException(
                status_code=400, 
                detail="Services sheet must contain 'service_id', 'price', and 'support_cost' columns"
            )

        # Sort by service_id to ensure correct order
        services_df = services_df.sort_values("service_id")
        price_data = services_df["price"].tolist()
        support_cost_data = services_df["support_cost"].tolist()

        # Extract orders data
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
        raise HTTPException(status_code=400, detail=f"Error processing Excel file: {str(e)}")

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

def combined_method(calculation_request: CalculationRequest, num_providers: int = 3) -> CombinedMethodResponse:
    """
    Implement the combined method as described in the README

    1. Solve subproblems for providers and IT companies using the probabilistic method (MOPSO)
    2. Apply heuristic improvement to the solution
    3. Return the final solution
    """
    # Step 1: Solve subproblems using MOPSO (probabilistic method)
    mopso = MOPSO(
        calculation_request,
        num_providers=num_providers,
        num_particles=30,
        max_iter=50
    )

    # Run the optimization
    archive = mopso.optimize(animate=False)

    # Get the best compromise solution
    best_solution = mopso.get_best_compromise_solution()

    # Step 2: Apply heuristic improvement
    improved_solution = apply_heuristic_improvement(mopso, best_solution)

    # Step 3: Generate detailed report
    report = mopso.generate_detailed_report(improved_solution)

    # Extract Pareto front for visualization
    pareto_front = [[p.current_objectives[0], p.current_objectives[1]] for p in archive]

    # Create and return the response
    return CombinedMethodResponse(
        service_allocation=report['service_allocation'],
        provider_analysis=report['provider_analysis'],
        financial_summary=report['financial_summary'],
        pareto_front=pareto_front
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

    return result

# Example usage with the CalculationRequest model
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
