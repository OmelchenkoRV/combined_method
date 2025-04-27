import numpy as np

class Particle:
    def __init__(self, k: int, m: int, bounds=(0,1)):
        # Position and velocity
        self.position = np.random.uniform(bounds[0], bounds[1], (k, m))
        self.velocity = np.random.uniform(-0.1, 0.1, (k, m))
        # Personal best
        self.pbest_position = self.position.copy()
        self.pbest_objectives = [float('-inf'), float('-inf')]
        # Current objectives
        self.current_objectives = [float('-inf'), float('-inf')]
        # Pareto ranking helpers
        self.rank = 0
        self.crowding_distance = 0

    def update_velocity(self, leader_pos, w: float, c1: float, c2: float):
        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.pbest_position - self.position)
            + c2 * r2 * (leader_pos - self.position)
        )
        self.velocity = np.clip(self.velocity, -0.1, 0.1)

    def update_position(self, bounds=(0,1)):
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1])