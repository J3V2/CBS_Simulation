import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time

# Constants
GRID_SIZE = 20  # Larger grid size for complex scenarios

# Pathfinding Helpers
def heuristic(pos, goal):
    """Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def get_neighbors(position):
    """Get valid neighbors."""
    neighbors = [(position[0] + dx, position[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    return [p for p in neighbors if 0 <= p[0] < GRID_SIZE and 0 <= p[1] < GRID_SIZE]

def reconstruct_path(came_from, current):
    """Reconstruct path from A*."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]

def a_star(start, goal, obstacles, constraints):
    """A* algorithm with constraints."""
    open_list = []
    closed_list = set()
    came_from = {}
    g_cost = {start: 0}
    f_cost = {start: heuristic(start, goal)}

    heapq.heappush(open_list, (f_cost[start], start))

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            return reconstruct_path(came_from, current)

        closed_list.add(current)

        for neighbor in get_neighbors(current):
            if neighbor in closed_list or neighbor in obstacles:
                continue

            # Check time-based constraints
            if any(constraint for constraint in constraints if constraint[1] == neighbor and constraint[2] == g_cost[current] + 1):
                continue

            tentative_g_cost = g_cost[current] + 1
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_cost[neighbor], neighbor))

    return []  # No path found

def detect_conflicts(paths):
    """Detect conflicts among agents."""
    conflicts = []
    max_time = max(len(path) for path in paths)
    for t in range(max_time):
        positions = {}
        for agent_id, path in enumerate(paths):
            if t < len(path):
                position = path[t]
                if position in positions:
                    conflicts.append((positions[position], agent_id, position, t))
                else:
                    positions[position] = agent_id
    return conflicts

# Traditional CBS
def cbs_traditional(starts, goals, obstacles):
    """Traditional CBS implementation."""
    paths = []
    constraints = []
    conflict_resolution_steps = 0

    for i in range(len(starts)):
        path = a_star(starts[i], goals[i], obstacles, constraints)
        paths.append(path)

    while True:
        conflicts = detect_conflicts(paths)
        if not conflicts:
            break

        # Resolve first conflict
        agent1, agent2, pos, time_step = conflicts[0]
        constraints.append((agent1, pos, time_step))
        constraints.append((agent2, pos, time_step))

        paths[agent1] = a_star(starts[agent1], goals[agent1], obstacles, constraints)
        paths[agent2] = a_star(starts[agent2], goals[agent2], obstacles, constraints)

        conflict_resolution_steps += 1

    return paths, conflict_resolution_steps

# Enhanced CBS with Priority
class CBS_Priority:
    def __init__(self, starts, goals, obstacles):
        self.starts = starts
        self.goals = goals
        self.obstacles = obstacles
        self.paths = []
        self.constraints = []

    def a_star(self, start, goal):
        """A* algorithm with constraints."""
        return a_star(start, goal, self.obstacles, self.constraints)

    def priority(self, agent_index):
        """Calculate priority based on proximity to the goal."""
        if self.paths[agent_index]:
            last_pos = self.paths[agent_index][-1]
            return heuristic(last_pos, self.goals[agent_index])
        return float('inf')

    def execute(self):
        """Enhanced CBS with priority system."""
        for i in range(len(self.starts)):
            self.paths.append(self.a_star(self.starts[i], self.goals[i]))

        conflict_resolution_steps = 0
        while True:
            conflicts = detect_conflicts(self.paths)
            if not conflicts:
                break

            # Resolve conflicts with priority
            agent1, agent2, pos, time_step = conflicts[0]
            if self.priority(agent1) < self.priority(agent2):
                self.constraints.append((agent2, pos, time_step))
                self.paths[agent2] = self.a_star(self.starts[agent2], self.goals[agent2])
            else:
                self.constraints.append((agent1, pos, time_step))
                self.paths[agent1] = self.a_star(self.starts[agent1], self.goals[agent1])

            conflict_resolution_steps += 1

        return self.paths, conflict_resolution_steps

# Visualization
def visualize_side_by_side(starts, goals, obstacles, solution_traditional, solution_enhanced, scenario_name):
    """Side-by-side visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def draw(ax, paths, title):
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)

        # Draw grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor="black", facecolor="white")
                ax.add_patch(rect)

        # Draw obstacles
        for ox, oy in obstacles:
            obstacle = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black")
            ax.add_patch(obstacle)

        # Draw start and goal positions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (start, goal) in enumerate(zip(starts, goals)):
            ax.text(*start, f'S{i+1}', ha='center', va='center', color='black', fontsize=10, weight='bold')
            ax.text(*goal, f'G{i+1}', ha='center', va='center', color='black', fontsize=10, weight='bold')

        # Draw paths
        for i, path in enumerate(paths):
            if len(path) > 1:
                ax.plot([p[0] for p in path], [p[1] for p in path], color=colors[i % len(colors)], lw=2)

    # Traditional CBS
    draw(axes[0], solution_traditional, "Traditional CBS")

    # Enhanced CBS
    draw(axes[1], solution_enhanced, "Enhanced CBS with Priority")

    plt.suptitle(scenario_name, fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

# Generate Random Scenario
def generate_random_scenario(grid_size, num_agents, num_obstacles):
    """Generate random starts, goals, and obstacles."""
    starts = []
    goals = []
    obstacles = set()

    while len(starts) < num_agents:
        start = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if start not in starts:
            starts.append(start)

    while len(goals) < num_agents:
        goal = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if goal not in goals and goal not in starts:
            goals.append(goal)

    while len(obstacles) < num_obstacles:
        obstacle = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if obstacle not in starts and obstacle not in goals:
            obstacles.add(obstacle)

    return starts, goals, list(obstacles)

# Main Test Scenarios
def run_complex_scenarios():
    scenarios = [
        {"name": "Bottleneck Challenge", "agents": 6, "obstacles": 12},
        {"name": "Crowded Goals", "agents": 8, "obstacles": 15},
        {"name": "Dense Obstacles", "agents": 4, "obstacles": 25},
    ]

    for scenario in scenarios:
        print(f"Running Scenario: {scenario['name']}")

        starts, goals, obstacles = generate_random_scenario(GRID_SIZE, scenario["agents"], scenario["obstacles"])

        # Run Traditional CBS
        solution_traditional, _ = cbs_traditional(starts, goals, obstacles)

        # Run Enhanced CBS
        cbs_priority = CBS_Priority(starts, goals, obstacles)
        solution_enhanced, _ = cbs_priority.execute()

        # Visualize
        visualize_side_by_side(starts, goals, obstacles, solution_traditional, solution_enhanced, scenario["name"])

if __name__ == "__main__":
    run_complex_scenarios()
