import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------- Global Parameters --------------------
GRID_SIZE = 20

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

# Global counters and cache for the two algorithms
A_STAR_CALLS_ORIGINAL = 0  # for Original CBS
A_STAR_CALLS_ENHANCED = 0  # for Enhanced CBS
GLOBAL_CACHE_ENHANCED = {}  # for hierarchical caching in Enhanced CBS

# ===========================
#  COMMON FUNCTIONS
# ===========================
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
    (This function is used in enhanced algorithm.)
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

# ===========================
#  ORIGINAL CBS
# ===========================
def a_star_orig(start, goal, obstacles, constraints):
    """
    A* algorithm with constraints.
    """
    global A_STAR_CALLS_ORIGINAL
    A_STAR_CALLS_ORIGINAL += 1

    open_list = []
    closed_set = set()
    came_from = {}
    g_cost = {start: 0}

    # use the heuristic.
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
            # Check constraints: if moving into neighbor at time (g_cost[current] + 1) is forbidden.
            if any((neighbor == pos and (g_cost[current] + 1) == t) for (pos, t) in constraints):
                continue
            next_g = g_cost[current] + 1
            new_cost = next_g  # No extra penalty is added.
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = new_cost
                f = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))
    return []  # No path found


def compute_paths_orig(starts, goals, obstacles, constraints):
    """Compute paths for all agents using Original CBS."""
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
    Original CBS (branching) implementation.
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
    Visualize the Original CBS solution on the given Axes.
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

# ===========================
#  ENHANCED CBS (with Prioritization, Advanced Tie Breaking, Hierarchical Caching, and CAT)
# ===========================
def a_star_enhanced(start, goal, obstacles, constraints, cat, other_agent_paths):
    """
    A* search for Enhanced CBS.
    Uses hierarchical caching to store full paths and subpaths.
    """
    global A_STAR_CALLS_ENHANCED, GLOBAL_CACHE_ENHANCED
    key = (start, goal, tuple(sorted(obstacles)), tuple(sorted(constraints)))
    if key in GLOBAL_CACHE_ENHANCED:
        return GLOBAL_CACHE_ENHANCED[key]
    A_STAR_CALLS_ENHANCED += 1

    open_list = []
    start_f = heuristic(start, goal)
    heapq.heappush(open_list, (start_f, 0, start, 0))
    came_from = {}
    g_cost = {start: 0}
    closed_set = set()

    while open_list:
        f_val, curr_g, current, curr_collisions = heapq.heappop(open_list)
        if current == goal:
            full_path = reconstruct_path(came_from, current)
            # Cache full path and every subpath.
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

def count_conflicts(paths):
    """
    Count total number of conflicts across all agents' paths.
    For each time step, for any cell occupied by more than one agent, add (n-1) to the count.
    """
    total_conflicts = 0
    max_time = max(len(p) for p in paths) if paths else 0
    for t in range(max_time):
        pos_count = {}
        for p in paths:
            if t < len(p):
                pos = p[t]
                pos_count[pos] = pos_count.get(pos, 0) + 1
        for count in pos_count.values():
            if count > 1:
                total_conflicts += count - 1
    return total_conflicts

class CBSNode_Enhanced:
    """
    A node in the CBS tree for Enhanced CBS.
    Tie-breaking: primary key = sum of path lengths; secondary key = conflict count.
    """
    __slots__ = ['constraints', 'paths', 'cost', 'conflict_count']

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, t)
        self.paths = paths  # list of agent paths
        self.cost = sum(len(p) for p in paths)
        self.conflict_count = count_conflicts(paths)

    def __lt__(self, other):
        if self.cost == other.cost:
            return self.conflict_count < other.conflict_count
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
        cat=cat,
        other_agent_paths=other_paths
    )

def cbs_enhanced(starts, goals, obstacles):
    """
    Enhanced CBS algorithm.
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
        # Advanced tie-breaking
        if heuristic(pos, goals[agent1]) <= heuristic(pos, goals[agent2]):
            favored_agent, other_agent = agent1, agent2
        else:
            favored_agent, other_agent = agent2, agent1

        # Branch for the other agent:
        child_constraints1 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints1[other_agent].append((pos, t))
        child_paths1 = current_node.paths[:]
        new_path_other = compute_paths_enhanced(other_agent, starts, goals, obstacles, child_constraints1, child_paths1)
        if new_path_other:
            child_paths1[other_agent] = new_path_other
            heapq.heappush(open_list, CBSNode_Enhanced(child_constraints1, child_paths1))

        # Branch for the favored agent:
        child_constraints2 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints2[favored_agent].append((pos, t))
        child_paths2 = current_node.paths[:]
        new_path_favored = compute_paths_enhanced(favored_agent, starts, goals, obstacles, child_constraints2, child_paths2)
        if new_path_favored:
            child_paths2[favored_agent] = new_path_favored
            heapq.heappush(open_list, CBSNode_Enhanced(child_constraints2, child_paths2))
    total_time = time.time() - start_time
    print(f"[Enhanced CBS] No solution after {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time

def visualize_solution_enhanced(ax, starts, goals, obstacles, solution, title="Enhanced CBS"):
    """
    Visualize the Enhanced CBS solution on the given Axes.
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

# ===========================
#  SIDE-BY-SIDE VISUALIZATION FUNCTION
# ===========================
def visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_enhanced,
                                     title1="Original CBS (Fixed Scenario)",
                                     title2="Enhanced CBS (Fixed Scenario)"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    visualize_solution_orig(ax1, starts, goals, obstacles, sol_orig, title=title1)
    visualize_solution_enhanced(ax2, starts, goals, obstacles, sol_enhanced, title=title2)
    plt.suptitle("Comparison of CBS Solutions", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

# ===========================
#  MAIN CODE (Fixed Scenario)
# ===========================
if __name__ == "__main__":
    # Fixed scenario: predetermined starts, goals, and obstacles.
    starts = [
            (1, 8), (8, 0), (19, 8), (6, 11), (7, 8),
            (3, 18), (8, 9), (3, 19), (12, 16), (19, 17),
            (2, 11), (8, 5), (13, 19), (8, 11), (4, 19),
            (4, 17), (6, 6), (9, 0), (12, 0), (11, 16),
            (6, 1), (9, 15), (12, 9), (1, 6), (4, 10),
            (0, 1), (5, 5), (19, 3), (1, 10), (12, 17)
    ]

    goals = [
            (10, 8), (16, 13), (14, 2), (11, 6), (13, 12),
            (16, 16), (13, 14), (14, 3), (14, 18), (15, 2),
            (6, 13), (10, 14), (9, 18), (13, 5), (5, 11),
            (11, 19), (2, 0), (18, 5), (2, 2), (13, 7),
            (6, 12), (15, 4), (9, 6), (9, 13), (1, 16),
            (4, 4), (13, 16), (12, 18), (13, 0), (9, 3)
    ]

    obstacles = [
                (0, 2), (0, 12), (1, 4), (1, 15), (2, 1),
                (2, 9), (3, 7), (4, 3), (4, 8), (4, 11),
                (5, 0), (5, 7), (7, 0), (7, 15), (7, 17),
                (8, 4), (8, 12), (8, 15), (10, 0), (10, 12),
                (10, 19), (12, 11), (12, 12), (13, 6), (13, 17),
                (16, 15), (16, 17), (18, 0), (18, 16), (19, 1)
    ]

    # --- Run Original CBS ---
    A_STAR_CALLS_ORIGINAL = 0
    print("Running Original CBS (Fixed Scenario)...")
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

    # --- Run Enhanced CBS ---
    A_STAR_CALLS_ENHANCED = 0
    GLOBAL_CACHE_ENHANCED = {}
    print("\nRunning Enhanced CBS (Fixed Scenario)...")
    start_time_enhanced = time.time()
    sol_enhanced, conflicts_enhanced, runtime_enhanced = cbs_enhanced(starts, goals, obstacles)
    runtime_enhanced = time.time() - start_time_enhanced
    solved_enhanced = sum(1 for path, goal in zip(sol_enhanced, goals) if path and path[-1] == goal)
    avg_len_enhanced = sum(len(path) for path in sol_enhanced) / len(sol_enhanced) if sol_enhanced else 0
    print("\n[Enhanced CBS] Results:")
    print(f"  Agents solved: {solved_enhanced}/{len(starts)}")
    print(f"  Runtime: {runtime_enhanced:.4f}s, Conflicts resolved: {conflicts_enhanced}")
    print(f"  Average Path Length: {avg_len_enhanced:.2f}")
    print(f"  A* Calls (Enhanced): {A_STAR_CALLS_ENHANCED}")

    # --- Side-by-Side Visualization ---
    visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_enhanced)
