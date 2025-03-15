import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
import sys

# -------------------- Basic Functions --------------------

GRID_SIZE = 20  # Grid size for scenarios


def heuristic(pos, goal):
    """Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def get_neighbors(position):
    """Return valid neighbors (up, down, left, right)."""
    neighbors = [(position[0] + dx, position[1] + dy)
                 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    return [p for p in neighbors if 0 <= p[0] < GRID_SIZE and 0 <= p[1] < GRID_SIZE]


def reconstruct_path(came_from, current):
    """Reconstruct the path from A*."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]


def a_star(start, goal, obstacles, constraints):
    """
    A* algorithm with constraints.
    'constraints' is a list of tuples (pos, time) that the agent must avoid.
    """
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
            if any((neighbor == pos and (g_cost[current] + 1) == t) for (pos, t) in constraints):
                continue
            tentative = g_cost[current] + 1
            if neighbor not in g_cost or tentative < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative
                f_cost[neighbor] = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_cost[neighbor], neighbor))
    return []  # No path found


def detect_conflicts(paths):
    """
    Detect the first conflict among agents.
    Returns (agent1, agent2, position, time) or None if no conflict exists.
    """
    max_time = max(len(p) for p in paths) if paths else 0
    for t in range(max_time):
        pos_dict = {}
        for i, path in enumerate(paths):
            if t < len(path):
                pos = path[t]
                if pos in pos_dict:
                    return (pos_dict[pos], i, pos, t)
                pos_dict[pos] = i
    return None


# -------------------- Original CBS (Branching) --------------------

class CBSNode:
    """A node in the CBS constraint tree."""
    __slots__ = ["constraints", "paths", "cost"]

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, t)
        self.paths = paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost


def compute_paths(starts, goals, obstacles, constraints):
    """Compute paths for all agents given per-agent constraints."""
    paths = []
    for i in range(len(starts)):
        paths.append(a_star(starts[i], goals[i], obstacles, constraints[i]))
    return paths


def cbs_original(starts, goals, obstacles):
    """
    Original CBS (branching) exactly as in Sharon et al. (2012),
    with a real-time timer printed every second.
    """
    root_constraints = {i: [] for i in range(len(starts))}
    root_paths = compute_paths(starts, goals, obstacles, root_constraints)
    root_node = CBSNode(root_constraints, root_paths)
    open_list = []
    heapq.heappush(open_list, root_node)
    conflict_resolution_steps = 0
    start_time = time.time()
    last_update = time.time()

    while open_list:
        node = heapq.heappop(open_list)
        conflict = detect_conflicts(node.paths)
        if conflict is None:
            total_time = time.time() - start_time
            print(
                f"\nOriginal CBS finished: Total time: {total_time:.5f}s, Conflicts resolved: {conflict_resolution_steps}")
            return node.paths, conflict_resolution_steps
        conflict_resolution_steps += 1
        if time.time() - last_update >= 1:
            elapsed = time.time() - start_time
            print(f"Running... Elapsed time: {elapsed:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
            last_update = time.time()
        agent1, agent2, pos, t = conflict
        # Branch for agent1:
        child_constraints1 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints1[agent1].append((pos, t))
        child_paths1 = list(node.paths)
        new_path1 = a_star(starts[agent1], goals[agent1], obstacles, child_constraints1[agent1])
        child_node1 = None
        if new_path1:
            child_paths1[agent1] = new_path1
            child_node1 = CBSNode(child_constraints1, child_paths1)
        # Branch for agent2:
        child_constraints2 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints2[agent2].append((pos, t))
        child_paths2 = list(node.paths)
        new_path2 = a_star(starts[agent2], goals[agent2], obstacles, child_constraints2[agent2])
        child_node2 = None
        if new_path2:
            child_paths2[agent2] = new_path2
            child_node2 = CBSNode(child_constraints2, child_paths2)
        if child_node1:
            heapq.heappush(open_list, child_node1)
        if child_node2:
            heapq.heappush(open_list, child_node2)
    total_time = time.time() - start_time
    print(f"\nOriginal CBS finished: Total time: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps


# -------------------- Visualization --------------------

def visualize_solution(starts, goals, obstacles, solution, scenario_name="Original CBS Solution"):
    """Display the solution from Original CBS in a single plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_title(scenario_name, fontsize=14)
    # Draw grid
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor="black", facecolor="white", lw=0.5)
            ax.add_patch(rect)
    # Draw obstacles
    for (x, y) in obstacles:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="black")
        ax.add_patch(rect)
    colors = ["green", "orange", "purple", "brown", "magenta", "cyan"]
    for i, (s, g) in enumerate(zip(starts, goals)):
        ax.scatter(s[0], s[1], c="blue", marker="o", s=300)
        ax.text(s[0], s[1], f"S{i + 1}", ha="center", va="center", color="white", fontsize=10)
        ax.scatter(g[0], g[1], c="red", marker="x", s=100)
        label = f"G{i + 1} ✓" if solution[i] and solution[i][-1] == g else f"G{i + 1} x"
        ax.text(g[0], g[1], label, ha="center", va="center", color="black", fontsize=10)
        if solution[i] and len(solution[i]) > 1:
            xs = [p[0] for p in solution[i]]
            ys = [p[1] for p in solution[i]]
            ax.plot(xs, ys, color=colors[i % len(colors)], lw=2)
    plt.suptitle(scenario_name, fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()


# -------------------- Random Scenario Generation --------------------

def generate_random_scenario(grid_size, num_agents, num_obstacles):
    """Generate a random scenario for demonstration."""
    starts = []
    goals = []
    obstacles_set = set()
    while len(starts) < num_agents:
        start = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if start not in starts:
            starts.append(start)
    while len(goals) < num_agents:
        goal = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if goal not in goals and goal not in starts:
            goals.append(goal)
    while len(obstacles_set) < num_obstacles:
        obs = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if obs not in starts and obs not in goals:
            obstacles_set.add(obs)
    return starts, goals, list(obstacles_set)


# -------------------- Main Code --------------------

if __name__ == "__main__":
    # Generate a random scenario.
    num_agents = 30
    num_obstacles = 15
    starts, goals, obstacles = generate_random_scenario(GRID_SIZE, num_agents, num_obstacles)

    # Run Original CBS (branching)
    start_time = time.time()
    sol_orig, conflicts_orig = cbs_original(starts, goals, obstacles)
    runtime_orig = time.time() - start_time
    solved_orig = sum(1 for path, goal in zip(sol_orig, goals) if path and path[-1] == goal)
    avg_len_orig = sum(len(path) for path in sol_orig) / len(sol_orig) if sol_orig else 0

    print("\nOriginal CBS Results:")
    print(f"  Agents solved: {solved_orig}/{num_agents}")
    print(f"  Runtime: {runtime_orig:.4f}s, Conflicts resolved: {conflicts_orig}")
    print(f"  Average Path Length: {avg_len_orig:.2f}\n")

    visualize_solution(starts, goals, obstacles, sol_orig, "CBS (Original) Solution")
