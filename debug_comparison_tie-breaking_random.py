import heapq
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------- Global Parameters --------------------
GRID_SIZE = 10
NUM_AGENTS = 10
NUM_OBSTACLES = 20

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

# Global counters for reporting
A_STAR_CALLS_ORIG = 0
A_STAR_CALLS_ENHANCED = 0


# -------------------- Random Scenario Generator --------------------
def generate_random_scenario(grid_size, num_agents, num_obstacles):
    """Generate a random scenario with unique start and goal positions and obstacles."""
    all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    random.shuffle(all_positions)
    starts = all_positions[:num_agents]
    goals = all_positions[num_agents:2 * num_agents]
    # Remove starts and goals from candidate obstacles
    remaining = [pos for pos in all_positions if pos not in starts and pos not in goals]
    obstacles = random.sample(remaining, num_obstacles)
    return starts, goals, obstacles


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
    Build a Conflict Avoidance Table (CAT): dict mapping time step -> set of positions.
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
    Count total number of conflicts across all agents' paths.
    For each time step, if a cell is occupied by more than one agent, add (n-1) to the count.
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


# -------------------- Original CBS Implementation --------------------
def a_star_orig(start, goal, obstacles, constraints, cat):
    """
    A* algorithm with constraints and a CAT.
    (No caching, tie-breaking, or prioritization.)
    """
    global A_STAR_CALLS_ORIG
    A_STAR_CALLS_ORIG += 1
    open_list = []
    closed_set = set()
    came_from = {}
    g_cost = {start: 0}
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
    return []


def compute_paths_orig(starts, goals, obstacles, constraints):
    """Compute paths for all agents using the original A* above."""
    paths = []
    for i in range(len(starts)):
        cat = build_conflict_avoidance_table([])  # initially empty
        paths.append(a_star_orig(starts[i], goals[i], obstacles, constraints[i], cat))
    return paths


class CBSNode_Orig:
    """
    CBS node for Original CBS.
    Priority is based solely on the total cost (sum of path lengths).
    """
    __slots__ = ["constraints", "paths", "cost"]

    def __init__(self, constraints, paths):
        self.constraints = constraints
        self.paths = paths
        self.cost = sum(len(p) for p in paths)

    def __lt__(self, other):
        return self.cost < other.cost


def cbs_original(starts, goals, obstacles):
    """
    Original CBS implementation.
    """
    n_agents = len(starts)
    root_constraints = {i: [] for i in range(n_agents)}
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
            print(f"\n[Original CBS] Finished: {total_time:.3f}s, Conflicts resolved: {conflict_resolution_steps}")
            return node.paths, conflict_resolution_steps, total_time
        conflict_resolution_steps += 1
        if time.time() - last_update >= 1:
            print(
                f"[Original CBS] Running... {time.time() - start_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
            last_update = time.time()
        agent1, agent2, pos, t = conflict
        # Branch for agent1:
        child_constraints1 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints1[agent1].append((pos, t))
        child_paths1 = list(node.paths)
        other_paths1 = [node.paths[i] for i in range(n_agents) if i != agent1]
        cat1 = build_conflict_avoidance_table(other_paths1)
        new_path1 = a_star_orig(starts[agent1], goals[agent1], obstacles, child_constraints1[agent1], cat1)
        if new_path1:
            child_paths1[agent1] = new_path1
            heapq.heappush(open_list, CBSNode_Orig(child_constraints1, child_paths1))
        # Branch for agent2:
        child_constraints2 = {a: list(c) for a, c in node.constraints.items()}
        child_constraints2[agent2].append((pos, t))
        child_paths2 = list(node.paths)
        other_paths2 = [node.paths[i] for i in range(n_agents) if i != agent2]
        cat2 = build_conflict_avoidance_table(other_paths2)
        new_path2 = a_star_orig(starts[agent2], goals[agent2], obstacles, child_constraints2[agent2], cat2)
        if new_path2:
            child_paths2[agent2] = new_path2
            heapq.heappush(open_list, CBSNode_Orig(child_constraints2, child_paths2))
    total_time = time.time() - start_time
    print(f"\n[Original CBS] Finished: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time


def visualize_solution_orig(ax, starts, goals, obstacles, solution, title="Original CBS"):
    """Visualize the solution computed by Original CBS."""
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor="black", facecolor="white", lw=0.5)
            ax.add_patch(rect)
    for (ox, oy) in obstacles:
        rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black")
        ax.add_patch(rect)
    for i, (s, g) in enumerate(zip(starts, goals)):
        ax.scatter(s[0], s[1], c="cyan", marker="o", s=START_MARKER_SIZE)
        ax.text(s[0], s[1], f"S{i + 1}", ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        ax.scatter(g[0], g[1], c="red", marker="x", s=GOAL_MARKER_SIZE)
        sol_i = solution[i] if i < len(solution) else []
        label = f"G{i + 1} ✓" if sol_i and sol_i[-1] == g else f"G{i + 1} x"
        ax.text(g[0], g[1], label, ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        if sol_i and len(sol_i) > 1:
            xs = [p[0] for p in sol_i]
            ys = [p[1] for p in sol_i]
            ax.plot(xs, ys, color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=PATH_LINE_WIDTH)


# -------------------- Enhanced CBS (Tie-Breaking Only) Implementation --------------------
def a_star_enhanced(start, goal, obstacles, constraints, cat):
    """
    A* search that respects per-agent constraints and uses a CAT.
    This version does not use caching or prioritization.
    """
    global A_STAR_CALLS_ENHANCED
    A_STAR_CALLS_ENHANCED += 1
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, 0))
    came_from = {}
    g_cost = {start: 0}
    closed_set = set()
    while open_list:
        f_val, curr_g, current, curr_collisions = heapq.heappop(open_list)
        if current == goal:
            return reconstruct_path(came_from, current)
        if current in closed_set:
            continue
        closed_set.add(current)
        next_g = curr_g + 1
        for neighbor in get_neighbors(current):
            if neighbor in obstacles:
                continue
            if any((neighbor == pos and (g_cost[current] + 1) == t) for (pos, t) in constraints):
                continue
            next_collisions = curr_collisions
            if next_g in cat and neighbor in cat[next_g]:
                next_collisions += 1
            cost_so_far = next_g + next_collisions * 2
            f_new = cost_so_far + heuristic(neighbor, goal)
            if neighbor not in g_cost or cost_so_far < g_cost[neighbor]:
                g_cost[neighbor] = cost_so_far
                came_from[neighbor] = current
                heapq.heappush(open_list, (f_new, next_g, neighbor, next_collisions))
    return []


def compute_paths_enhanced(agent_id, starts, goals, obstacles, constraints, solution_so_far):
    """
    Recompute path for one agent using the CAT.
    """
    other_paths = [p for i, p in enumerate(solution_so_far) if i != agent_id]
    cat = build_conflict_avoidance_table(other_paths)
    return a_star_enhanced(starts[agent_id], goals[agent_id], obstacles, constraints[agent_id], cat)


class CBSNode_Enhanced:
    """
    CBS node for Enhanced CBS (Tie-Breaking Only).
    Priority is based on:
      - Primary key: total cost (sum of path lengths)
      - Secondary key: total conflict count (computed by count_conflicts)
    """
    __slots__ = ['constraints', 'paths', 'cost', 'conflict_count']

    def __init__(self, constraints, paths):
        self.constraints = constraints
        self.paths = paths
        self.cost = sum(len(p) for p in paths)
        self.conflict_count = count_conflicts(paths)

    def __lt__(self, other):
        if self.cost == other.cost:
            return self.conflict_count < other.conflict_count
        return self.cost < other.cost


def cbs_enhanced(starts, goals, obstacles):
    """
    Enhanced CBS (Tie-Breaking Only) implementation.
    When a conflict is detected, the algorithm chooses the favored agent
    (the one whose goal is closer to the conflict point) and branches accordingly.
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
            print(f"\n[Enhanced CBS] Finished: {total_time:.3f}s, Conflicts resolved: {conflict_resolution_steps}")
            return current_node.paths, conflict_resolution_steps, total_time
        conflict_resolution_steps += 1
        if time.time() - last_update >= 1:
            print(
                f"[Enhanced CBS] Running... {time.time() - start_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
            last_update = time.time()
        agent1, agent2, pos, t = conflict
        # Tie-Breaking: choose the agent whose goal is closer to the conflict point.
        if heuristic(pos, goals[agent1]) <= heuristic(pos, goals[agent2]):
            favored_agent, other_agent = agent1, agent2
        else:
            favored_agent, other_agent = agent2, agent1
        # Branch for agent1 (first branch for non-favored, then for favored).
        child_constraints1 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints1[agent1].append((pos, t))
        child_paths1 = list(current_node.paths)
        new_path1 = compute_paths_enhanced(agent1, starts, goals, obstacles, child_constraints1, child_paths1)
        if new_path1:
            child_paths1[agent1] = new_path1
            heapq.heappush(open_list, CBSNode_Enhanced(child_constraints1, child_paths1))
        child_constraints2 = {a: list(c) for a, c in current_node.constraints.items()}
        child_constraints2[agent2].append((pos, t))
        child_paths2 = list(current_node.paths)
        new_path2 = compute_paths_enhanced(agent2, starts, goals, obstacles, child_constraints2, child_paths2)
        if new_path2:
            child_paths2[agent2] = new_path2
            heapq.heappush(open_list, CBSNode_Enhanced(child_constraints2, child_paths2))
    total_time = time.time() - start_time
    print(f"\n[Enhanced CBS] No solution: {total_time:.2f}s, Conflicts resolved: {conflict_resolution_steps}")
    return [], conflict_resolution_steps, total_time


def visualize_solution(ax, starts, goals, obstacles, solution, title):
    """Visualize a CBS solution on the given Axes."""
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor="black", facecolor="white", lw=0.5)
            ax.add_patch(rect)
    for (ox, oy) in obstacles:
        rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black")
        ax.add_patch(rect)
    for i, (s, g) in enumerate(zip(starts, goals)):
        ax.scatter(s[0], s[1], c="cyan", marker="o", s=START_MARKER_SIZE)
        ax.text(s[0], s[1], f"S{i + 1}", ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        ax.scatter(g[0], g[1], c="red", marker="x", s=GOAL_MARKER_SIZE)
        sol_i = solution[i] if i < len(solution) else []
        label = f"G{i + 1} ✓" if sol_i and sol_i[-1] == g else f"G{i + 1} x"
        ax.text(g[0], g[1], label, ha="center", va="center", color="black", fontsize=LABEL_FONT_SIZE)
        if sol_i and len(sol_i) > 1:
            xs = [p[0] for p in sol_i]
            ys = [p[1] for p in sol_i]
            ax.plot(xs, ys, color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=PATH_LINE_WIDTH)


def visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_enh, title1, title2):
    """Visualize two solutions side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    visualize_solution(ax1, starts, goals, obstacles, sol_orig, title1)
    visualize_solution(ax2, starts, goals, obstacles, sol_enh, title2)
    plt.suptitle("CBS Solutions Comparison", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()


# -------------------- MAIN CODE --------------------
if __name__ == "__main__":
    # Generate a random scenario.
    starts, goals, obstacles = generate_random_scenario(GRID_SIZE, NUM_AGENTS, NUM_OBSTACLES)
    print("Random scenario generated.")
    print("Starts:", starts)
    print("Goals:", goals)
    print("Obstacles:", obstacles)

    # Reset A* call counters.
    A_STAR_CALLS_ORIG = 0
    A_STAR_CALLS_ENHANCED = 0

    # Run Original CBS.
    print("\nRunning Original CBS...")
    start_time = time.time()
    sol_orig, conflicts_orig, runtime_orig = cbs_original(starts, goals, obstacles)
    runtime_orig = time.time() - start_time
    solved_orig = sum(1 for path, goal in zip(sol_orig, goals) if path and path[-1] == goal)
    avg_len_orig = sum(len(path) for path in sol_orig) / len(sol_orig) if sol_orig else 0
    print(f"[Original CBS] Agents solved: {solved_orig}/{len(starts)}")
    print(f"[Original CBS] Runtime: {runtime_orig:.4f}s, Conflicts resolved: {conflicts_orig}")
    print(f"[Original CBS] Average Path Length: {avg_len_orig:.2f}")
    print(f"[Original CBS] A* Calls: {A_STAR_CALLS_ORIG}")

    # Run Enhanced CBS (Tie-Breaking Only).
    print("\nRunning Enhanced CBS (Tie-Breaking Only)...")
    start_time = time.time()
    sol_enh, conflicts_enh, runtime_enh = cbs_enhanced(starts, goals, obstacles)
    runtime_enh = time.time() - start_time
    solved_enh = sum(1 for path, goal in zip(sol_enh, goals) if path and path[-1] == goal)
    avg_len_enh = sum(len(path) for path in sol_enh) / len(sol_enh) if sol_enh else 0
    print(f"[Enhanced CBS] Agents solved: {solved_enh}/{len(starts)}")
    print(f"[Enhanced CBS] Runtime: {runtime_enh:.4f}s, Conflicts resolved: {conflicts_enh}")
    print(f"[Enhanced CBS] Average Path Length: {avg_len_enh:.2f}")
    print(f"[Enhanced CBS] A* Calls: {A_STAR_CALLS_ENHANCED}")

    # Visualize side by side.
    visualize_solutions_side_by_side(starts, goals, obstacles, sol_orig, sol_enh,
                                     title1="Original CBS", title2="Enhanced CBS (Tie-Breaking Only)")
