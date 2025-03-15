import heapq
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

# -------------------- Global Parameters --------------------
GRID_SIZE = 20  # Grid size for scenarios

# Global agent colors for consistent visualization
AGENT_COLORS = ["red", "orange", "purple", "brown", "magenta",
                "cyan", "blue", "green", "olive", "navy"]

# UI Constants
START_MARKER_SIZE = 300
GOAL_MARKER_SIZE = 300
LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 14  # Title font size for each subplot
PATH_LINE_WIDTH = 2

# Global counters to track low-level search calls
A_STAR_CALLS_ORIGINAL = 0
A_STAR_CALLS_ENHANCED = 0

# Global cache for Enhanced CBS only (hierarchical caching)
# Keys: (start, goal, obstacles, constraints) and subkeys for intermediate nodes.
GLOBAL_CACHE_ENHANCED = {}

# -------------------- Basic Functions --------------------
def heuristic(pos, goal):
    """Manhattan distance heuristic (used in Original CBS)."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def manhattan_distance(a, b):
    """Manhattan distance between two points a and b."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(position):
    """Return valid neighbors (up, down, left, right)."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(position[0] + dx, position[1] + dy)
            for dx, dy in directions if 0 <= position[0] + dx < GRID_SIZE and 0 <= position[1] + dy < GRID_SIZE]

def reconstruct_path(came_from, current):
    """Reconstruct the path from A*."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]

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

# -------------------- Build Conflict Avoidance Table (CAT) --------------------
def build_conflict_avoidance_table(paths):
    """
    Build a Conflict Avoidance Table (CAT): dict mapping time_step -> set of positions.
    """
    cat = {}
    if not paths:
        return cat
    max_len = max(len(p) for p in paths)
    for t in range(max_len):
        cat[t] = set()
        for path in paths:
            if t < len(path):
                cat[t].add(path[t])
    return cat

# -------------------- ORIGINAL CBS (Branching) WITHOUT Caching --------------------
def a_star(start, goal, obstacles, constraints):
    """
    A* algorithm with constraints.
    'constraints' is a list of tuples (pos, time) that the agent must avoid.
    (No caching is used in Original CBS.)
    """
    global A_STAR_CALLS_ORIGINAL
    A_STAR_CALLS_ORIGINAL += 1

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

def compute_paths(starts, goals, obstacles, constraints):
    """Compute paths for all agents given per-agent constraints."""
    paths = []
    for i in range(len(starts)):
        paths.append(a_star(starts[i], goals[i], obstacles, constraints[i]))
    return paths

class CBSNode:
    """A node in the CBS constraint tree for Original CBS."""
    __slots__ = ["constraints", "paths", "cost"]

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, time)
        self.paths = paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost

def cbs_original(starts, goals, obstacles):
    """
    Original CBS (branching) exactly as in Sharon et al. (2012).
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
            print(f"\n[Original CBS] Finished: Total time: {total_time:.5f}s, Conflicts resolved: {conflict_resolution_steps}")
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
    print(f"\n[Original CBS] Finished: Total time: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time

def visualize_solution_orig(ax, starts, goals, obstacles, solution, title="Original CBS"):
    """
    Draw the Original CBS solution on the provided Axes.
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
        ax.text(s[0], s[1], f"S{i+1}", ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        ax.scatter(g[0], g[1], c="red", marker="x", s=GOAL_MARKER_SIZE)
        sol = solution[i] if i < len(solution) else []
        label = f"G{i+1} ✓" if sol and sol[-1] == g else f"G{i+1} x"
        ax.text(g[0], g[1], label, ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        if sol and len(sol) > 1:
            xs = [p[0] for p in sol]
            ys = [p[1] for p in sol]
            ax.plot(xs, ys, color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=PATH_LINE_WIDTH)

# -------------------- ENHANCED CBS (Priority + CAT) WITH HIERARCHICAL CACHING --------------------
def a_star_with_cat(start, goal, obstacles, constraints, cat, other_agent_paths):
    """
    A* that respects per-agent (pos, t) constraints and uses a CAT to penalize collisions.
    Hierarchical caching is applied: after computing a full path result,
    all subpaths (from each intermediate node to the goal) are cached.
    """
    global A_STAR_CALLS_ENHANCED, GLOBAL_CACHE_ENHANCED
    # Create a key for the full query.
    key = (start, goal, tuple(sorted(obstacles)), tuple(sorted(constraints)))
    if key in GLOBAL_CACHE_ENHANCED:
        return GLOBAL_CACHE_ENHANCED[key]
    A_STAR_CALLS_ENHANCED += 1

    open_list = []
    start_f = manhattan_distance(start, goal)
    heapq.heappush(open_list, (start_f, 0, start, 0))
    came_from = {}
    g_cost = {start: 0}
    closed_set = set()

    while open_list:
        f_val, curr_g, current, curr_collisions = heapq.heappop(open_list)
        if current == goal:
            full_path = reconstruct_path(came_from, current)
            # Hierarchical caching: cache full path and every subpath from each intermediate node.
            for i in range(len(full_path)):
                sub_start = full_path[i]
                sub_key = (sub_start, goal, tuple(sorted(obstacles)), tuple(sorted(constraints)))
                sub_path = full_path[i:]
                GLOBAL_CACHE_ENHANCED[sub_key] = sub_path
            GLOBAL_CACHE_ENHANCED[key] = full_path
            return full_path
        if current in closed_set:
            continue
        closed_set.add(current)
        next_g = curr_g + 1
        for neighbor in get_neighbors(current):
            if neighbor in obstacles:
                continue
            if any((cpos == neighbor and ctime == next_g) for (cpos, ctime) in constraints):
                continue
            next_collisions = curr_collisions
            if next_g in cat and neighbor in cat[next_g]:
                next_collisions += 1
            cost_so_far = next_g + next_collisions * 2
            f_new = cost_so_far + manhattan_distance(neighbor, goal)
            if (neighbor not in g_cost) or (cost_so_far < g_cost[neighbor]):
                g_cost[neighbor] = cost_so_far
                came_from[neighbor] = current
                heapq.heappush(open_list, (f_new, next_g, neighbor, next_collisions))
    result = []
    GLOBAL_CACHE_ENHANCED[key] = result
    return result

class CBSNodePCAT:
    """
    A node in the CBS tree for Enhanced CBS (Priority + CAT).
    """
    __slots__ = ['constraints', 'paths', 'cost']

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, t)
        self.paths = paths              # list of agent paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost

def compute_path_for_agent_pcat(agent_id, starts, goals, obstacles, constraints, solution_so_far):
    """
    Recompute path for one agent using the CAT.
    """
    other_paths = [p for i, p in enumerate(solution_so_far) if i != agent_id]
    cat = build_conflict_avoidance_table(other_paths)
    return a_star_with_cat(
        start=starts[agent_id],
        goal=goals[agent_id],
        obstacles=obstacles,
        constraints=constraints[agent_id],
        cat=cat,
        other_agent_paths=other_paths
    )

def cbs_priority_cat(starts, goals, obstacles):
    """
    Enhanced CBS (Priority + CAT) algorithm.
    """
    n_agents = len(starts)
    root_constraints = {i: [] for i in range(n_agents)}
    root_paths = []
    for i in range(n_agents):
        path_i = compute_path_for_agent_pcat(i, starts, goals, obstacles, root_constraints, root_paths)
        root_paths.append(path_i)
    root_node = CBSNodePCAT(root_constraints, root_paths)
    open_list = []
    heapq.heappush(open_list, root_node)
    conflict_resolution_steps = 0
    start_time = time.time()
    last_update = time.time()

    while open_list:
        current_node = heapq.heappop(open_list)
        conflict = detect_conflicts(current_node.paths)
        if conflict is None:
            total_time = time.time() - start_time
            print(f"\n[Enhanced CBS] Finished: Total time: {total_time:.3f}s, Conflicts resolved: {conflict_resolution_steps}")
            return current_node.paths, conflict_resolution_steps, total_time
        conflict_resolution_steps += 1
        if time.time() - last_update >= 1:
            elapsed = time.time() - start_time
            print(f"[Enhanced CBS] Running... Elapsed: {elapsed:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
            last_update = time.time()
        agent1, agent2, pos, t = conflict
        if manhattan_distance(pos, goals[agent1]) <= manhattan_distance(pos, goals[agent2]):
            favored_agent, other_agent = agent1, agent2
        else:
            favored_agent, other_agent = agent2, agent1

        # Branch for the other agent:
        child_constraints1 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints1[other_agent].append((pos, t))
        child_paths1 = current_node.paths[:]
        new_path_other = compute_path_for_agent_pcat(other_agent, starts, goals, obstacles, child_constraints1, child_paths1)
        if new_path_other:
            child_paths1[other_agent] = new_path_other
            node1 = CBSNodePCAT(child_constraints1, child_paths1)
            heapq.heappush(open_list, node1)

        # Branch for the favored agent:
        child_constraints2 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints2[favored_agent].append((pos, t))
        child_paths2 = current_node.paths[:]
        new_path_favored = compute_path_for_agent_pcat(favored_agent, starts, goals, obstacles, child_constraints2, child_paths2)
        if new_path_favored:
            child_paths2[favored_agent] = new_path_favored
            node2 = CBSNodePCAT(child_constraints2, child_paths2)
            heapq.heappush(open_list, node2)
    total_time = time.time() - start_time
    print(f"[Enhanced CBS] No solution after {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time

def visualize_solution_pcat(ax, starts, goals, obstacles, solution, title="Enhanced CBS"):
    """
    Draw the Enhanced CBS solution on the provided Axes.
    """
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                     edgecolor="black", facecolor="white", lw=0.5)
            ax.add_patch(rect)
    for (ox, oy) in obstacles:
        rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color="black")
        ax.add_patch(rect)
    for i, (s, g) in enumerate(zip(starts, goals)):
        ax.scatter(s[0], s[1], c="cyan", marker="o", s=START_MARKER_SIZE)
        ax.text(s[0], s[1], f"S{i+1}", color="black", ha="center", va="center", fontsize=LABEL_FONT_SIZE)
        ax.scatter(g[0], g[1], c="red", marker="x", s=GOAL_MARKER_SIZE)
        sol = solution[i] if i < len(solution) else []
        label = f"G{i+1} ✓" if sol and sol[-1] == g else f"G{i+1} x"
        ax.text(g[0], g[1], label, color="black", ha="center", va="center", fontsize=LABEL_FONT_SIZE)
        if sol:
            xs = [p[0] for p in sol]
            ys = [p[1] for p in sol]
            ax.plot(xs, ys, color=AGENT_COLORS[i % len(AGENT_COLORS)], linewidth=PATH_LINE_WIDTH)

def visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_pcat, title1="Original CBS", title2="Enhanced CBS"):
    """
    Display the two solutions side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    visualize_solution_pcat(ax1, starts, goals, obstacles, sol_orig, title1)
    visualize_solution_pcat(ax2, starts, goals, obstacles, sol_pcat, title2)
    plt.suptitle("CBS Solutions Comparison", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

# -------------------- MAIN CODE --------------------
if __name__ == "__main__":
    num_agents = 10
    num_obstacles = 0

    # Reset counters and cache for Enhanced CBS only.
    A_STAR_CALLS_ORIGINAL = 0
    A_STAR_CALLS_ENHANCED = 0
    GLOBAL_CACHE_ENHANCED = {}

    # Generate a random scenario (same for both algorithms)
    starts, goals, obstacles = generate_random_scenario(GRID_SIZE, num_agents, num_obstacles)

    # Run Original CBS (without caching)
    print("Running Original CBS...")
    start_time_orig = time.time()
    sol_orig, conflicts_orig, runtime_orig = cbs_original(starts, goals, obstacles)
    runtime_orig = time.time() - start_time_orig
    solved_orig = sum(1 for path, goal in zip(sol_orig, goals) if path and path[-1] == goal)
    avg_len_orig = sum(len(path) for path in sol_orig) / len(sol_orig) if sol_orig else 0

    # Run Enhanced CBS (with hierarchical caching)
    print("\nRunning CBS-Priority...")
    start_time_pcat = time.time()
    sol_pcat, conflicts_pcat, runtime_pcat = cbs_priority_cat(starts, goals, obstacles)
    runtime_pcat = time.time() - start_time_pcat
    solved_pcat = sum(1 for path, goal in zip(sol_pcat, goals) if path and path[-1] == goal)
    avg_len_pcat = sum(len(path) for path in sol_pcat) / len(sol_pcat) if sol_pcat else 0

    # Print metrics for Original CBS
    print("\n[Original CBS] Results:")
    print(f"  Agents solved: {solved_orig}/{num_agents}")
    print(f"  Runtime: {runtime_orig:.4f}s, Conflicts resolved: {conflicts_orig}")
    print(f"  Average Path Length: {avg_len_orig:.2f}")
    print(f"  A* Calls (Original): {A_STAR_CALLS_ORIGINAL}")

    # Print metrics for Enhanced CBS
    print("\n[Enhanced CBS] Results:")
    print(f"  Agents solved: {solved_pcat}/{num_agents}")
    print(f"  Runtime: {runtime_pcat:.4f}s, Conflicts resolved: {conflicts_pcat}")
    print(f"  Average Path Length: {avg_len_pcat:.2f}")
    print(f"  A* Calls (Enhanced): {A_STAR_CALLS_ENHANCED}")

    # Visualize the two solutions side by side
    visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_pcat,
                                     title1="Original CBS", title2="Enhanced CBS")
