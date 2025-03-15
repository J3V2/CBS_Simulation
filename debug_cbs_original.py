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

# Global counter for low-level A* calls (for reporting)
A_STAR_CALLS = 0

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
    Build a Conflict Avoidance Table (CAT): a dict mapping time step -> set of positions.
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

# -------------------- ORIGINAL CBS (with CAT in A*) --------------------
def a_star(start, goal, obstacles, constraints, cat):
    """
    A* algorithm with constraints and an optional Conflict Avoidance Table.
    If a conflict (collision) is predicted (via the CAT), a penalty is added.
    Caching, tie-breaking, and prioritization enhancements are removed.
    """
    global A_STAR_CALLS
    A_STAR_CALLS += 1

    open_list = []
    closed_set = set()
    came_from = {}
    g_cost = {start: 0}

    # If CAT indicates a conflict at time 0, add a penalty.
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
            # Respect constraints: do not allow a move if (neighbor, time) is forbidden.
            if any((neighbor == pos and (g_cost[current] + 1) == t) for (pos, t) in constraints):
                continue
            tentative_g = g_cost[current] + 1
            penalty = 2 if (cat is not None and tentative_g in cat and neighbor in cat[tentative_g]) else 0
            new_cost = tentative_g + penalty
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = new_cost
                f = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))
    return []  # No path found

def compute_paths(starts, goals, obstacles, constraints):
    """Compute paths for all agents using the A* defined above."""
    paths = []
    for i in range(len(starts)):
        # Initially, no other agents' paths exist.
        cat = build_conflict_avoidance_table([])
        paths.append(a_star(starts[i], goals[i], obstacles, constraints[i], cat))
    return paths

# -------------------- CBS NODE --------------------
class CBSNode:
    """
    A node in the CBS tree.
    This is the original CBS node which uses only the total cost (sum of path lengths).
    """
    __slots__ = ["constraints", "paths", "cost"]

    def __init__(self, constraints, paths):
        self.constraints = constraints  # dict: agent -> list of (pos, time)
        self.paths = paths              # list of agent paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost

def cbs_original(starts, goals, obstacles):
    """
    Original CBS (branching) implementation with Conflict Avoidance Table (CAT)
    integrated into the low-level A*. Enhancements from Enhanced CBS are removed.
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
        other_paths1 = [node.paths[i] for i in range(len(node.paths)) if i != agent1]
        cat1 = build_conflict_avoidance_table(other_paths1)
        new_path1 = a_star(starts[agent1], goals[agent1], obstacles, child_constraints1[agent1], cat1)
        if new_path1:
            child_paths1[agent1] = new_path1
            heapq.heappush(open_list, CBSNode(child_constraints1, child_paths1))

        # Branch for agent2:
        child_constraints2 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints2[agent2].append((pos, t))
        child_paths2 = list(node.paths)
        other_paths2 = [node.paths[i] for i in range(len(node.paths)) if i != agent2]
        cat2 = build_conflict_avoidance_table(other_paths2)
        new_path2 = a_star(starts[agent2], goals[agent2], obstacles, child_constraints2[agent2], cat2)
        if new_path2:
            child_paths2[agent2] = new_path2
            heapq.heappush(open_list, CBSNode(child_constraints2, child_paths2))
    total_time = time.time() - start_time
    print(f"\n[Original CBS] Finished: Total time: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time

# -------------------- Visualization --------------------
def visualize_solution(ax, starts, goals, obstacles, solution, title="Original CBS (Fixed Scenario)"):
    """
    Draw the CBS solution on the provided Axes.
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
    for (ox, oy) in obstacles:
        rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black")
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

    # Reset global counter.
    A_STAR_CALLS = 0

    print("Running Original CBS (Fixed Scenario)...")
    start_time = time.time()
    sol, conflicts, runtime = cbs_original(starts, goals, obstacles)
    runtime = time.time() - start_time
    solved = sum(1 for path, goal in zip(sol, goals) if path and path[-1] == goal)
    avg_len = sum(len(path) for path in sol) / len(sol) if sol else 0

    print("\n[Original CBS] Results:")
    print(f"  Agents solved: {solved}/{len(starts)}")
    print(f"  Runtime: {runtime:.4f}s, Conflicts resolved: {conflicts}")
    print(f"  Average Path Length: {avg_len:.2f}")
    print(f"  A* Calls (Original): {A_STAR_CALLS}")

    # Visualize the solution
    fig, ax = plt.subplots(figsize=(8, 8))
    visualize_solution(ax, starts, goals, obstacles, sol, title="Original CBS (Fixed Scenario)")
    plt.tight_layout()
    plt.show()
