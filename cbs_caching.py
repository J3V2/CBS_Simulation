import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------- Global Parameters --------------------
GRID_SIZE = 10

# Global agent colors for consistent visualization
AGENT_COLORS = ["red", "orange", "purple", "brown", "magenta",
                "cyan", "blue", "green", "olive", "navy",
                "lime", "teal", "gold", "silver", "maroon",
                "indigo", "violet", "chartreuse", "coral", "turquoise",
                "chocolate", "salmon", "plum", "orchid", "skyblue",
                "khaki", "sienna", "peru", "slateblue", "pink"]

# UI Constants
START_MARKER_SIZE = 300
GOAL_MARKER_SIZE = 300
LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 14
PATH_LINE_WIDTH = 2

# Global counters and cache for Enhanced CBS (hierarchical caching)
A_STAR_CALLS_ENHANCED = 0
GLOBAL_CACHE_ENHANCED = {}


# -------------------- Basic Functions --------------------
def heuristic(pos, goal):
    """Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def get_neighbors(position):
    """Return valid neighbors (up, down, left, right)."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(position[0] + dx, position[1] + dy)
            for dx, dy in directions
            if 0 <= position[0] + dx < GRID_SIZE and 0 <= position[1] + dy < GRID_SIZE]

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

# -------------------- ENHANCED CBS WITH CACHING ONLY --------------------
def a_star_enhanced(start, goal, obstacles, constraints, cat):
    """
    A* search that respects per-agent constraints and uses a CAT to penalize collisions.
    Hierarchical caching is enabled: once a full path is computed, all its subpaths are cached.
    """
    global A_STAR_CALLS_ENHANCED, GLOBAL_CACHE_ENHANCED
    key = (start, goal, tuple(sorted(obstacles)), tuple(sorted(constraints)))
    if key in GLOBAL_CACHE_ENHANCED:
        return GLOBAL_CACHE_ENHANCED[key]
    A_STAR_CALLS_ENHANCED += 1

    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, 0))
    came_from = {}
    g_cost = {start: 0}
    closed_set = set()

    while open_list:
        f_val, curr_g, current, curr_collisions = heapq.heappop(open_list)
        if current == goal:
            full_path = reconstruct_path(came_from, current)
            # Cache the full path and every subpath from each intermediate node.
            for i in range(len(full_path)):
                sub_start = full_path[i]
                sub_key = (sub_start, goal, tuple(sorted(obstacles)), tuple(sorted(constraints)))
                GLOBAL_CACHE_ENHANCED[sub_key] = full_path[i:]
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
            f_new = cost_so_far + heuristic(neighbor, goal)
            if (neighbor not in g_cost) or (cost_so_far < g_cost[neighbor]):
                g_cost[neighbor] = cost_so_far
                came_from[neighbor] = current
                heapq.heappush(open_list, (f_new, next_g, neighbor, next_collisions))
    GLOBAL_CACHE_ENHANCED[key] = []
    return []

class CBSNode_Enhanced:
    """
    A node in the CBS tree for Enhanced CBS using caching only.
    Priority is based solely on the total cost (sum of path lengths).
    """
    __slots__ = ['constraints', 'paths', 'cost']

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, t)
        self.paths = paths              # list of agent paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost

def compute_paths_enhanced(agent_id, starts, goals, obstacles, constraints, solution_so_far):
    """
    Recompute path for one agent using the CAT.
    """
    other_paths = [p for i, p in enumerate(solution_so_far) if i != agent_id]
    cat = build_conflict_avoidance_table(other_paths)
    return a_star_enhanced(
        start=starts[agent_id],
        goal=goals[agent_id],
        obstacles=obstacles,
        constraints=constraints[agent_id],
        cat=cat
    )

def cbs_enhanced(starts, goals, obstacles):
    """
    Enhanced CBS algorithm that uses caching (hierarchical caching) but
    removes both the prioritization for favored agents and the tie-breaking mechanism.
    In this version, when a conflict is detected, the algorithm branches for both agents in a fixed order.
    """
    n_agents = len(starts)
    root_constraints = {i: [] for i in range(n_agents)}
    root_paths = []
    for i in range(n_agents):
        path_i = compute_paths_enhanced(i, starts, goals, obstacles, root_constraints, root_paths)
        root_paths.append(path_i)
    root_node = CBSNode_Enhanced(root_constraints, root_paths)
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

        # Branch for agent1:
        child_constraints1 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints1[agent1].append((pos, t))
        child_paths1 = list(current_node.paths)
        new_path1 = compute_paths_enhanced(agent1, starts, goals, obstacles, child_constraints1, child_paths1)
        if new_path1:
            child_paths1[agent1] = new_path1
            heapq.heappush(open_list, CBSNode_Enhanced(child_constraints1, child_paths1))

        # Branch for agent2:
        child_constraints2 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints2[agent2].append((pos, t))
        child_paths2 = list(current_node.paths)
        new_path2 = compute_paths_enhanced(agent2, starts, goals, obstacles, child_constraints2, child_paths2)
        if new_path2:
            child_paths2[agent2] = new_path2
            heapq.heappush(open_list, CBSNode_Enhanced(child_constraints2, child_paths2))
    total_time = time.time() - start_time
    print(f"[Enhanced CBS] No solution after {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time

# -------------------- Visualization --------------------
def visualize_solution_enhanced(ax, starts, goals, obstacles, solution, title="Enhanced CBS with Caching Only"):
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

# -------------------- MAIN CODE (Fixed Scenario) --------------------
if __name__ == "__main__":
    # Fixed scenario: predetermined starts, goals, and obstacles.
    starts = [(0, 9), (9, 7), (7, 9), (2, 6), (2, 8)]
    goals = [(8, 9), (6, 3), (2, 1), (4, 1), (2, 2)]
    obstacles = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4),
                 (1, 9), (2, 4), (2, 5), (3, 5), (3, 6),
                 (3, 9), (4, 2), (4, 4), (5, 4), (5, 5),
                 (5, 7), (5, 8), (6, 1), (6, 2), (6, 9),
                 (7, 0), (7, 1), (7, 8), (8, 1), (8, 2),
                 (8, 3), (8, 5), (9, 0), (9, 5), (9, 6)]
    # Reset global counters and cache.
    A_STAR_CALLS_ENHANCED = 0
    GLOBAL_CACHE_ENHANCED = {}

    print("Running Enhanced CBS (Fixed Scenario, Caching Only)...")
    start_time_pcat = time.time()
    sol_enhanced, conflicts_enhanced, runtime_enhanced = cbs_enhanced(starts, goals, obstacles)
    runtime_enhanced = time.time() - start_time_pcat
    solved_enhanced = sum(1 for path, goal in zip(sol_enhanced, goals) if path and path[-1] == goal)
    avg_len_enhanced = sum(len(path) for path in sol_enhanced) / len(sol_enhanced) if sol_enhanced else 0

    # Print metrics for Enhanced CBS
    print("\n[Enhanced CBS] Results:")
    print(f"  Agents solved: {solved_enhanced}/{len(starts)}")
    print(f"  Runtime: {runtime_enhanced:.4f}s, Conflicts resolved: {conflicts_enhanced}")
    print(f"  Average Path Length: {avg_len_enhanced:.2f}")
    print(f"  A* Calls (Enhanced): {A_STAR_CALLS_ENHANCED}")

    # Visualize the solution
    fig, ax = plt.subplots(figsize=(8, 8))
    visualize_solution_enhanced(ax, starts, goals, obstacles, sol_enhanced,
                                title="Enhanced CBS (Caching Only, Fixed Scenario)")
    plt.tight_layout()
    plt.show()
