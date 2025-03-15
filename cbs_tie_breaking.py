import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------- Global Parameters --------------------
GRID_SIZE = 10  # Using a 20x20 grid

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

# Global counter to track low-level A* search calls
A_STAR_CALLS_ORIGINAL = 0

# -------------------- Basic Functions --------------------
def heuristic(pos, goal):
    """Manhattan distance heuristic (used in Original CBS)."""
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

def count_conflicts(paths):
    """
    Count the total number of conflicts across all agents' paths.
    For each time step, if a cell is occupied by more than one agent, add (n-1) to the conflict count.
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

# -------------------- ORIGINAL CBS WITH CAT in A* --------------------
def a_star(start, goal, obstacles, constraints, cat=None):
    """
    A* algorithm with constraints and an optional Conflict Avoidance Table.
    If a conflict (i.e. collision) is predicted at a time step (from the CAT),
    a penalty is added to the cost.
    """
    global A_STAR_CALLS_ORIGINAL
    A_STAR_CALLS_ORIGINAL += 1

    open_list = []
    closed_set = set()
    came_from = {}
    g_cost = {start: 0}

    # Starting penalty: if CAT is provided and a collision is predicted at time 0.
    start_penalty = 0
    if cat is not None and 0 in cat and start in cat[0]:
        start_penalty = 2

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
            # Check if the neighbor violates any constraint (position, time)
            if any((neighbor == pos and (g_cost[current] + 1) == t) for (pos, t) in constraints):
                continue
            next_g = g_cost[current] + 1
            penalty = 0
            if cat is not None:
                # If the CAT shows a collision at the next time step, add a penalty.
                if next_g in cat and neighbor in cat[next_g]:
                    penalty = 2
            new_cost = next_g + penalty
            tentative = new_cost
            if neighbor not in g_cost or tentative < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative
                f = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))
    return []  # No path found

def compute_paths_orig(starts, goals, obstacles, constraints):
    """Compute paths for all agents given per-agent constraints."""
    paths = []
    for i in range(len(starts)):
        # For the initial planning, we have no other paths—pass an empty CAT.
        cat = build_conflict_avoidance_table([])
        paths.append(a_star(starts[i], goals[i], obstacles, constraints[i], cat=cat))
    return paths

# -------------------- CBS NODE WITH TIE-BREAKING --------------------
class CBSNode:
    """A node in the CBS constraint tree for Original CBS with tie-breaking.
       Primary key: total cost (sum of path lengths)
       Secondary key: total conflict count
    """
    __slots__ = ["constraints", "paths", "cost", "conflict_count"]

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, time)
        self.paths = paths
        self.cost = sum(len(p) for p in paths)
        self.conflict_count = count_conflicts(paths)

    def __lt__(self, other):
        # Primary comparison on cost; if equal, use conflict_count.
        if self.cost == other.cost:
            return self.conflict_count < other.conflict_count
        return self.cost < other.cost

def cbs_original(starts, goals, obstacles):
    """
    Original CBS (branching) implementation with Conflict Avoidance Table (CAT)
    integrated into the low-level A* and with a tie-breaking mechanism.
    """
    root_constraints = {i: [] for i in range(len(starts))}
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
            print(f"\n[Original CBS] Finished: Total time: {total_time:.5f}s, Conflicts resolved: {conflict_resolution_steps}")
            return node.paths, conflict_resolution_steps, total_time
        conflict_resolution_steps += 1
        if time.time() - last_update >= 1:
            elapsed = time.time() - start_time
            print(f"[Original CBS] Running... Elapsed: {elapsed:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
            last_update = time.time()
        agent1, agent2, pos, t = conflict

        # --- Branch for agent1 ---
        child_constraints1 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints1[agent1].append((pos, t))
        child_paths1 = list(node.paths)
        # Build a CAT for agent1 using the paths of all other agents.
        other_paths1 = [node.paths[i] for i in range(len(node.paths)) if i != agent1]
        cat1 = build_conflict_avoidance_table(other_paths1)
        new_path1 = a_star(starts[agent1], goals[agent1], obstacles, child_constraints1[agent1], cat=cat1)
        if new_path1:
            child_paths1[agent1] = new_path1
            heapq.heappush(open_list, CBSNode(child_constraints1, child_paths1))

        # --- Branch for agent2 ---
        child_constraints2 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints2[agent2].append((pos, t))
        child_paths2 = list(node.paths)
        # Build a CAT for agent2 using the paths of all other agents.
        other_paths2 = [node.paths[i] for i in range(len(node.paths)) if i != agent2]
        cat2 = build_conflict_avoidance_table(other_paths2)
        new_path2 = a_star(starts[agent2], goals[agent2], obstacles, child_constraints2[agent2], cat=cat2)
        if new_path2:
            child_paths2[agent2] = new_path2
            heapq.heappush(open_list, CBSNode(child_constraints2, child_paths2))
    total_time = time.time() - start_time
    print(f"\n[Original CBS] Finished: Total time: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time

# -------------------- Visualization --------------------
def visualize_solution_orig(ax, starts, goals, obstacles, solution, title="Original CBS with Tie-breaking"):
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

# -------------------- MAIN CODE (Fixed Scenario) --------------------
if __name__ == "__main__":
    # Fixed scenario: 30 agents, predetermined starts, goals, and obstacles.
    starts = [(0, 9), (9, 7), (7, 9), (2, 6), (2, 8)]
    goals = [(8, 9), (6, 3), (2, 1), (4, 1), (2, 2)]
    obstacles = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4),
                 (1, 9), (2, 4), (2, 5), (3, 5), (3, 6),
                 (3, 9), (4, 2), (4, 4), (5, 4), (5, 5),
                 (5, 7), (5, 8), (6, 1), (6, 2), (6, 9),
                 (7, 0), (7, 1), (7, 8), (8, 1), (8, 2),
                 (8, 3), (8, 5), (9, 0), (9, 5), (9, 6)]

    # Reset the global A* counter.
    A_STAR_CALLS_ORIGINAL = 0

    print("Running Original CBS (Fixed Scenario)...")
    start_time_orig = time.time()
    sol_orig, conflicts_orig, runtime_orig = cbs_original(starts, goals, obstacles)
    runtime_orig = time.time() - start_time_orig
    solved_orig = sum(1 for path, goal in zip(sol_orig, goals) if path and path[-1] == goal)
    avg_len_orig = sum(len(path) for path in sol_orig) / len(sol_orig) if sol_orig else 0

    # Print metrics for Original CBS
    print("\n[Original CBS] Results:")
    print(f"  Agents solved: {solved_orig}/{len(starts)}")
    print(f"  Runtime: {runtime_orig:.4f}s, Conflicts resolved: {conflicts_orig}")
    print(f"  Average Path Length: {avg_len_orig:.2f}")
    print(f"  A* Calls (Original): {A_STAR_CALLS_ORIGINAL}")

    # Visualize the solution
    fig, ax = plt.subplots(figsize=(8, 8))
    visualize_solution_orig(ax, starts, goals, obstacles, sol_orig, title="Original CBS with Advance Tie-Breaking (Fixed Scenario)")
    plt.tight_layout()
    plt.show()
