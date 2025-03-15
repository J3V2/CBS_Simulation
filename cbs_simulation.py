import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time

# Constants
GRID_SIZE = 40  # Larger grid size for complex scenarios

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
            if any(constraint for constraint in constraints if
                   constraint[1] == neighbor and constraint[2] == g_cost[current] + 1):
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

# Visualization
def visualize_enhanced(starts, goals, obstacles, solution_enhanced, scenario_name):
    """Visualization for Enhanced CBS."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"{scenario_name} - Enhanced CBS", fontsize=14)

    # Draw grid
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor="black", facecolor="white")
            ax.add_patch(rect)

    # Draw obstacles
    for ox, oy in obstacles:
        obstacle = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black")
        ax.add_patch(obstacle)

    # Colors list for paths
    colors = [
        'aqua', 'black', 'blue', 'blueviolet', 'brown', 'cadetblue', 'chartreuse',
        'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue',
        'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
        'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
        'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey',
        'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey',
        'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod',
        'gray', 'green', 'greenyellow', 'grey', 'hotpink', 'indianred', 'indigo',
        'lawngreen', 'lightcoral', 'lightgreen', 'lightseagreen', 'lightslategray',
        'lightslategrey', 'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue',
        'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navy',
        'olive', 'olivedrab', 'orange', 'orangered', 'peru', 'purple', 'rebeccapurple',
        'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
        'seagreen', 'sienna', 'slateblue', 'slategray', 'slategrey', 'springgreen',
        'steelblue', 'tan', 'teal', 'tomato', 'turquoise', 'yellow', 'yellowgreen'
    ]

    # Draw start and goal positions with a checkmark if solved.
    for i, (start, goal) in enumerate(zip(starts, goals)):
        ax.text(*start, f'S{i + 1}', ha='center', va='center', color='black', fontsize=10, weight='bold')
        if i < len(solution_enhanced) and solution_enhanced[i] and solution_enhanced[i][-1] == goal:
            goal_text = f'G{i + 1} ✓'
        else:
            goal_text = f'G{i + 1}'
        ax.text(*goal, goal_text, ha='center', va='center', color='black', fontsize=10, weight='bold')

    # Draw paths for Enhanced CBS
    for i, path in enumerate(solution_enhanced):
        if len(path) > 1:
            ax.plot([p[0] for p in path], [p[1] for p in path], color=colors[i % len(colors)], lw=2)

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
        {"name": "Bottleneck Challenge", "agents": 20, "obstacles": 50},
        # {"name": "Bottleneck Challenge 2", "agents": 10, "obstacles": 30},
        # {"name": "Bottleneck Challenge 3", "agents": 30, "obstacles": 50},
    ]

    traditional_metrics = {"runtime": [], "conflicts": [], "path_length": []}

    for scenario in scenarios:
        print(f"Running Scenario: {scenario['name']}")

        starts, goals, obstacles = generate_random_scenario(GRID_SIZE, scenario["agents"], scenario["obstacles"])

        # Run Traditional CBS

        start_time = time.time()
        solution_traditional, conflicts_traditional = cbs_traditional(starts, goals, obstacles)
        runtime_traditional = time.time() - start_time
        traditional_metrics["runtime"].append(runtime_traditional)
        traditional_metrics["conflicts"].append(conflicts_traditional)
        traditional_metrics["path_length"].append(sum(len(path) for path in solution_traditional) / len(solution_traditional))

        traditional_solved = sum(1 for path, goal in zip(solution_traditional, goals) if path and path[-1] == goal)
        traditional_unsolved = scenario["agents"] - traditional_solved

        print("\nResults:")
        print(f"Traditional CBS - Agents solved: {traditional_solved}/{scenario['agents']} (Unsolved: {traditional_unsolved})")
        print(
            f"Traditional CBS - Avg Runtime: {sum(traditional_metrics['runtime']) / len(traditional_metrics['runtime']):.4f}s, "
            f"Avg Conflicts: {sum(traditional_metrics['conflicts']) / len(traditional_metrics['conflicts']):.2f}, "
            f"Avg Path Length: {sum(traditional_metrics['path_length']) / len(traditional_metrics['path_length']):.2f}")

        # Visualize only Traditional CBS
        visualize_enhanced(starts, goals, obstacles, solution_traditional, scenario["name"])

if __name__ == "__main__":
    run_complex_scenarios()
