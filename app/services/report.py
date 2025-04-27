import numpy as np
from .mopso import MOPSO, Particle

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

    report = {}

    # Service selection matrix
    report['service_selection'] = solution.position.copy()

    # Count how many services are selected
    selected_services = np.sum(solution.position > 0.5)
    print(f"Number of selected services: {selected_services}")

    # Average daily orders
    avg_orders = np.mean(self.daily_orders, axis=0)
    report['avg_daily_orders'] = avg_orders

    # Create service allocation report
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

    report['service_allocation'] = service_allocation

    # Generate provider analysis
    provider_analysis = []

    for j in range(self.m):
        services = []
        total_revenue = 0
        total_cost = 0

        for i in range(self.k):
            if solution.position[i, j] > 0.5:  # Service i is allocated to provider j
                avg_service_orders = avg_orders[i]
                service_revenue = self.p[i, j] * (1 - self.r[i, j]) * avg_service_orders
                service_cost = self.s[i, j] * avg_service_orders
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

    # Calculate financial summary directly from the solution
    it_profit = 0
    provider_profit = 0

    for i in range(self.k):
        for j in range(self.m):
            if solution.position[i, j] > 0.5:
                # IT company profit
                service_profit = self.d[i, j] * (1 - self.r[i, j]) * avg_orders[i]
                it_profit += service_profit

    # Sum up provider profits
    for provider in provider_analysis:
        provider_profit += provider['total_profit']

    report['financial_summary'] = {
        'it_company_profit': it_profit,
        'provider_profit': provider_profit,
        'system_total_profit': it_profit + provider_profit
    }

    return report