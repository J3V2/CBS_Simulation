import heapq
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------- Global Parameters --------------------
GRID_SIZE = 25  # Grid dimensions
NUM_AGENTS = 0  # Number of agents
NUM_OBSTACLES = 0  # Number of obstacles (set to 0 for a free grid)

# Global agent colors for visualization
AGENT_COLORS = [
    "red", "orange", "purple", "brown", "magenta",
    "cyan", "blue", "green", "olive", "navy",
    "lime", "teal", "gold", "silver", "maroon",
    "indigo", "violet", "chartreuse", "coral", "turquoise",
    "chocolate", "salmon", "plum", "orchid", "skyblue",
    "khaki", "sienna", "peru", "slateblue", "pink"
]

# UI Constants
START_MARKER_SIZE = 300
GOAL_MARKER_SIZE = 300
LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 14
PATH_LINE_WIDTH = 2

# Global counter for low-level A* calls in Original CBS
A_STAR_CALLS_ORIGINAL = 0


# ===========================
#  RANDOM SCENARIO GENERATION
# ===========================
def generate_random_scenario(num_agents, grid_size, num_obstacles):
    """
    Generate a random scenario.
    Returns:
      - starts: list of start positions (tuples)
      - goals: list of goal positions (tuples)
      - obstacles: list of obstacle positions (tuples)
    """
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    if len(cells) < 2 * num_agents + num_obstacles:
        raise ValueError("Too many agents and obstacles for the grid size.")
    chosen = random.sample(cells, 2 * num_agents + num_obstacles)
    starts = chosen[:num_agents]
    goals = chosen[num_agents:2 * num_agents]
    obstacles = chosen[2 * num_agents:]
    return starts, goals, obstacles


# ===========================
#  COMMON FUNCTIONS
# ===========================
def heuristic(pos, goal):
    """Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def get_neighbors(position):
    """Return valid neighbors (up, down, left, right)."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (position[0] + dx, position[1] + dy)
        for dx, dy in directions
        if 0 <= position[0] + dx < GRID_SIZE and 0 <= position[1] + dy < GRID_SIZE
    ]


def reconstruct_path(came_from, current):
    """Reconstruct the path from A* search."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]


def detect_conflicts(paths):
    """
    Detect the first conflict among agents.
    Returns a tuple (agent1, agent2, position, time) or None if no conflict exists.
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


# ===========================
#  ORIGINAL CBS ALGORITHM
# ===========================
def a_star_orig(start, goal, obstacles, constraints):
    """
    Low-level A* search with constraints for Original CBS.
    """
    global A_STAR_CALLS_ORIGINAL
    A_STAR_CALLS_ORIGINAL += 1

    open_list = []
    closed_set = set()
    came_from = {}
    g_cost = {start: 0}
    f_cost = {start: g_cost[start] + heuristic(start, goal)}
    heapq.heappush(open_list, (f_cost[start], start))

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            return reconstruct_path(came_from, current)
        closed_set.add(current)
        for neighbor in get_neighbors(current):
            if neighbor in closed_set or neighbor in obstacles:
                continue
            # Check constraints: disallow moves into forbidden cells at specific times.
            if any((neighbor == pos and (g_cost[current] + 1) == t) for (pos, t) in constraints):
                continue
            next_g = g_cost[current] + 1
            new_cost = next_g  # No extra penalty is used.
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = new_cost
                f = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))
    return []  # No path found


def compute_paths_orig(starts, goals, obstacles, constraints):
    """Compute paths for all agents using the Original CBS approach."""
    paths = []
    for i in range(len(starts)):
        paths.append(a_star_orig(starts[i], goals[i], obstacles, constraints[i]))
    return paths


class CBSNode_Orig:
    """A node in the CBS tree for Original CBS."""
    __slots__ = ["constraints", "paths", "cost"]

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, time)
        self.paths = paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost


def cbs_original(starts, goals, obstacles):
    """
    The main Conflict-Based Search (CBS) algorithm for Original CBS.
    """
    root_constraints = {i: [] for i in range(len(starts))}
    root_paths = compute_paths_orig(starts, goals, obstacles, root_constraints)
    root_node = CBSNode_Orig(root_constraints, root_paths)
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
                f"\n[Original CBS] Finished: Total time: {total_time:.5f}s, Conflicts resolved: {conflict_resolution_steps}")
            return node.paths, conflict_resolution_steps, total_time
        conflict_resolution_steps += 1
        if time.time() - last_update >= 1:
            elapsed = time.time() - start_time
            print(f"[Original CBS] Running... Elapsed: {elapsed:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
            last_update = time.time()
        agent1, agent2, pos, t = conflict

        # Branch for agent1:
        child_constraints1 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints1[agent1].append((pos, t))
        child_paths1 = list(node.paths)
        new_path1 = a_star_orig(starts[agent1], goals[agent1], obstacles, child_constraints1[agent1])
        if new_path1:
            child_paths1[agent1] = new_path1
            heapq.heappush(open_list, CBSNode_Orig(child_constraints1, child_paths1))

        # Branch for agent2:
        child_constraints2 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints2[agent2].append((pos, t))
        child_paths2 = list(node.paths)
        new_path2 = a_star_orig(starts[agent2], goals[agent2], obstacles, child_constraints2[agent2])
        if new_path2:
            child_paths2[agent2] = new_path2
            heapq.heappush(open_list, CBSNode_Orig(child_constraints2, child_paths2))

    total_time = time.time() - start_time
    print(f"\n[Original CBS] Finished: Total time: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time


def visualize_solution_orig(ax, starts, goals, obstacles, solution, title="Original CBS"):
    """
    Visualize the CBS solution on a matplotlib Axes.
    """
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                     edgecolor="black", facecolor="white", lw=0.5)
            ax.add_patch(rect)
    for (x, y) in obstacles:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="black")
        ax.add_patch(rect)
    for i, (s, g) in enumerate(zip(starts, goals)):
        ax.scatter(s[0], s[1], c="cyan", marker="o", s=START_MARKER_SIZE)
        ax.text(s[0], s[1], f"S{i + 1}", ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        ax.scatter(g[0], g[1], c="red", marker="x", s=GOAL_MARKER_SIZE)
        sol = solution[i] if i < len(solution) else []
        label = f"G{i + 1} ✓" if sol and sol[-1] == g else f"G{i + 1} x"
        ax.text(g[0], g[1], label, ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        if sol and len(sol) > 1:
            xs = [p[0] for p in sol]
            ys = [p[1] for p in sol]
            ax.plot(xs, ys, color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=PATH_LINE_WIDTH)


# ===========================
#  MAIN CODE (Original CBS)
# ===========================
if __name__ == "__main__":
    # Generate a random scenario.
    starts, goals, obstacles = generate_random_scenario(NUM_AGENTS, GRID_SIZE, NUM_OBSTACLES)

    print("Randomly Generated Scenario:")
    print("Starts:", starts)
    print("Goals:", goals)
    print("Obstacles:", obstacles)

    # --- Run Original CBS ---
    A_STAR_CALLS_ORIGINAL = 0  # Reset counter
    print("\nRunning Original CBS (Random Scenario)...")
    start_time_orig = time.time()
    sol_orig, conflicts_orig, runtime_orig = cbs_original(starts, goals, obstacles)
    runtime_orig = time.time() - start_time_orig
    solved_orig = sum(1 for path, goal in zip(sol_orig, goals) if path and path[-1] == goal)
    avg_len_orig = sum(len(path) for path in sol_orig) / len(sol_orig) if sol_orig else 0
    print("\n[Original CBS] Results:")
    print(f"  Agents solved: {solved_orig}/{len(starts)}")
    print(f"  Runtime: {runtime_orig:.4f}s, Conflicts resolved: {conflicts_orig}")
    print(f"  Average Path Length: {avg_len_orig:.2f}")
    print(f"  A* Calls (Original): {A_STAR_CALLS_ORIGINAL}")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    visualize_solution_orig(ax, starts, goals, obstacles, sol_orig, title="Original CBS")
    plt.show()
