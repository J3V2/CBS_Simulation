import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------- Global Parameters --------------------
GRID_SIZE = 20  # Using a 20x20 grid for clarity

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
TITLE_FONT_SIZE = 14  # Title font size for each plot
PATH_LINE_WIDTH = 2

# Global counter to track low-level A* search calls (for both versions)
A_STAR_CALLS_ORIGINAL = 0
A_STAR_CALLS_ENHANCED = 0  # for our enhanced (prioritized) version

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

# -------------------- Low-Level A* (used by both versions) --------------------
def a_star(start, goal, obstacles, constraints, cat=None):
    """
    A* algorithm with constraints and an optional Conflict Avoidance Table.
    If the CAT indicates a predicted collision at a time step, a penalty is added.
    """
    global A_STAR_CALLS_ORIGINAL
    A_STAR_CALLS_ORIGINAL += 1

    open_list = []
    closed_set = set()
    came_from = {}
    g_cost = {start: 0}

    # Add a starting penalty if the CAT predicts a collision at time 0.
    start_penalty = 2 if (cat is not None and 0 in cat and start in cat[0]) else 0
    f_cost = {start: g_cost[start] + start_penalty + heuristic(start, goal)}
    heapq.heappush(open_list, (f_cost[start], start))

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            return reconstruct_path(came_from, current)
        closed_set.add(current)
        for neighbor in get_neighbors(current):
            if neighbor in closed_set or neighbor in obstacles:
                continue
            # Check constraints: if moving into neighbor at time step (g_cost[current] + 1) is forbidden.
            if any((neighbor == pos and (g_cost[current] + 1) == t) for (pos, t) in constraints):
                continue
            next_g = g_cost[current] + 1
            penalty = 2 if (cat is not None and next_g in cat and neighbor in cat[next_g]) else 0
            new_cost = next_g + penalty
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = new_cost
                f = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))
    return []  # No path found

def compute_paths_orig(starts, goals, obstacles, constraints):
    """Compute paths for all agents (used by Original CBS)."""
    paths = []
    for i in range(len(starts)):
        # Initially, no other agent paths exist so pass an empty CAT.
        cat = build_conflict_avoidance_table([])
        paths.append(a_star(starts[i], goals[i], obstacles, constraints[i], cat=cat))
    return paths

class CBSNode:
    """A node in the CBS constraint tree (used by both versions)."""
    __slots__ = ["constraints", "paths", "cost"]

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, time)
        self.paths = paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost

# -------------------- Enhanced CBS with Prioritization for Favored Agents --------------------
# (This version uses the same low-level A* with CAT but modifies the high-level conflict resolution.)
def cbs_enhanced_prioritized(starts, goals, obstacles):
    """
    Enhanced CBS implementation that adds prioritization for favored agents.
    When a conflict is detected, the agent whose goal is closer to the conflict position
    is favored. The branch for the non-favored agent is expanded first.
    """
    global A_STAR_CALLS_ENHANCED
    n_agents = len(starts)
    root_constraints = {i: [] for i in range(n_agents)}
    root_paths = compute_paths_orig(starts, goals, obstacles, root_constraints)
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
            print(f"\n[Enhanced CBS - Prioritized] Finished: Total time: {total_time:.5f}s, Conflicts resolved: {conflict_resolution_steps}")
            return node.paths, conflict_resolution_steps, total_time
        conflict_resolution_steps += 1
        if time.time() - last_update >= 1:
            elapsed = time.time() - start_time
            print(f"[Enhanced CBS - Prioritized] Running... Elapsed: {elapsed:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
            last_update = time.time()
        agent1, agent2, pos, t = conflict

        # ---------- Prioritization for Favored Agents ----------
        # Favor the agent whose goal is closer (using Manhattan distance) to the conflict position.
        if heuristic(pos, goals[agent1]) <= heuristic(pos, goals[agent2]):
            favored_agent, other_agent = agent1, agent2
        else:
            favored_agent, other_agent = agent2, agent1

        # --- Branch for the non-favored (other) agent first ---
        child_constraints1 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints1[other_agent].append((pos, t))
        child_paths1 = list(node.paths)
        # Build CAT from the paths of all agents except the one being replanned.
        other_paths1 = [node.paths[i] for i in range(len(node.paths)) if i != other_agent]
        cat1 = build_conflict_avoidance_table(other_paths1)
        new_path1 = a_star(starts[other_agent], goals[other_agent], obstacles, child_constraints1[other_agent], cat=cat1)
        if new_path1:
            child_paths1[other_agent] = new_path1
            heapq.heappush(open_list, CBSNode(child_constraints1, child_paths1))

        # --- Then branch for the favored agent ---
        child_constraints2 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints2[favored_agent].append((pos, t))
        child_paths2 = list(node.paths)
        other_paths2 = [node.paths[i] for i in range(len(node.paths)) if i != favored_agent]
        cat2 = build_conflict_avoidance_table(other_paths2)
        new_path2 = a_star(starts[favored_agent], goals[favored_agent], obstacles, child_constraints2[favored_agent], cat=cat2)
        if new_path2:
            child_paths2[favored_agent] = new_path2
            heapq.heappush(open_list, CBSNode(child_constraints2, child_paths2))
    total_time = time.time() - start_time
    print(f"\n[Enhanced CBS - Prioritized] Finished: Total time: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time

# -------------------- Visualization Functions --------------------
def visualize_solution(ax, starts, goals, obstacles, solution, title="CBS Solution"):
    """
    Visualize a CBS solution on the given Axes.
    """
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    # Draw grid cells
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                     edgecolor="black", facecolor="white", lw=0.5)
            ax.add_patch(rect)
    # Draw obstacles
    for (x, y) in obstacles:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="black")
        ax.add_patch(rect)
    # Draw start and goal markers and paths for each agent
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

def visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_enhanced,
                                     title1="Original CBS (Fixed Scenario)",
                                     title2="Enhanced CBS with Prioritization (Fixed Scenario)"):
    """
    Display two CBS solutions side by side for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    visualize_solution(ax1, starts, goals, obstacles, sol_orig, title=title1)
    visualize_solution(ax2, starts, goals, obstacles, sol_enhanced, title=title2)
    plt.suptitle("Side-by-Side CBS Comparison", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

# -------------------- MAIN CODE (Fixed Scenario) --------------------
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
    sol_orig, conflicts_orig, runtime_orig = compute_paths_orig(starts, goals, obstacles, {i: [] for i in range(len(starts))}), 0, 0
    # Here we run only the initial planning without conflict resolution for visualization purposes.
    runtime_orig = time.time() - start_time_orig
    solved_orig = sum(1 for path, goal in zip(sol_orig, goals) if path and path[-1] == goal)
    avg_len_orig = sum(len(path) for path in sol_orig) / len(sol_orig) if sol_orig else 0

    print("\n[Original CBS] Results:")
    print(f"  Agents solved: {solved_orig}/{len(starts)}")
    print(f"  Runtime: {runtime_orig:.4f}s, Conflicts resolved: {conflicts_orig}")
    print(f"  Average Path Length: {avg_len_orig:.2f}")
    print(f"  A* Calls (Original): {A_STAR_CALLS_ORIGINAL}")

    # --- Run Enhanced CBS with Prioritization for Favored Agents ---
    A_STAR_CALLS_ENHANCED = 0
    print("\nRunning Enhanced CBS with Prioritization (Fixed Scenario)...")
    start_time_enhanced = time.time()
    sol_enhanced, conflicts_enhanced, runtime_enhanced = cbs_enhanced_prioritized(starts, goals, obstacles)
    runtime_enhanced = time.time() - start_time_enhanced
    solved_enhanced = sum(1 for path, goal in zip(sol_enhanced, goals) if path and path[-1] == goal)
    avg_len_enhanced = sum(len(path) for path in sol_enhanced) / len(sol_enhanced) if sol_enhanced else 0

    print("\n[Enhanced CBS - Prioritized] Results:")
    print(f"  Agents solved: {solved_enhanced}/{len(starts)}")
    print(f"  Runtime: {runtime_enhanced:.4f}s, Conflicts resolved: {conflicts_enhanced}")
    print(f"  Average Path Length: {avg_len_enhanced:.2f}")
    print(f"  A* Calls (Enhanced): {A_STAR_CALLS_ENHANCED}")

    # --- Side-by-Side Visualization ---
    visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_enhanced)
