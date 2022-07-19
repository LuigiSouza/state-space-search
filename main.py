from __future__ import annotations
import matplotlib.pyplot as plt
import time

SQRT_2 = 1.4


class Edge:
    def __init__(self, origin: "SubGoal", destiny: "SubGoal", weight: int):
        self.origin = origin
        self.destiny = destiny
        self.weight = weight


class SubGoal:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.edges: dict[str, Edge] = {}

    def generate_nodes(self, vertices: dict, grid: list[list]):
        max_top = self.__clearence((self.x, self.y), 0, -1, grid)
        max_bot = self.__clearence((self.x, self.y), 0, +1, grid)
        max_left = self.__clearence((self.x, self.y), -1, 0, grid)
        max_right = self.__clearence((self.x, self.y), +1, 0, grid)
        # Explorar diagonais
        # top right
        self.__expand_diagonal(+1, -1, max_right, max_top, grid)
        # bottom right
        self.__expand_diagonal(+1, +1, max_right, max_bot, grid)
        # bottom left
        self.__expand_diagonal(-1, +1, max_left, max_bot, grid)
        # top left
        self.__expand_diagonal(-1, -1, max_left, max_top, grid)

    def __expand_diagonal(
        self, dir_x: int, dir_y: int, horizontal: int, vertical: int, grid: list[list]
    ):
        curr = (self.x, self.y)
        while self.__has_diagonal(curr, dir_x, dir_y, grid):
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            horizontal = self.__clearence(curr, dir_x, 0, grid, horizontal)
            vertical = self.__clearence(curr, 0, dir_y, grid, vertical)
            if type(grid[curr[1]][curr[0]]) == SubGoal:
                self.add_edge(grid[curr[1]][curr[0]])

    def __has_diagonal(
        self,
        origin: tuple[int, int],
        dir_x: int,
        dir_y: int,
        grid: list[list],
    ):
        x = origin[0]
        y = origin[1]
        target = (x + dir_x, y + dir_y)

        if (
            x <= 0
            or 0 >= y
            or target[1] >= len(grid)
            or target[0] >= len(grid[target[1]])
        ):
            return False
        if (
            grid[target[1]][target[0]] == 0
            or grid[target[1]][x] == 0
            or 0 == grid[y][target[0]]
        ):
            return False
        return True

    def a_graph_search(self, destiny: "SubGoal", skip: list[str] = []):
        key = str(self.x) + "," + str(self.y)
        opened_nodes: dict[str, tuple[int, SubGoal, str]] = {key: (0, self, None)}
        closed_nodes: dict[str, tuple[int, SubGoal, str]] = {}
        while opened_nodes:
            lowest = min(opened_nodes.values(), key=lambda t: t[0])[1]
            key = str(lowest.x) + "," + str(lowest.y)
            curr_weight, curr_subgoal, father_key = opened_nodes.pop(key)
            if curr_subgoal == destiny:
                path: list[SubGoal] = []
                path.append(curr_subgoal)
                father = closed_nodes[father_key]
                total_weight = curr_subgoal.edges[father_key].weight
                while father[2] != None:
                    path.append(father[1])
                    total_weight += father[1].edges[father[2]].weight
                    father = closed_nodes[father[2]]
                path.append(father[1])
                return path, total_weight
            closed_nodes[key] = (curr_weight, curr_subgoal, father_key)
            for edge in curr_subgoal.edges:
                curr_edge = curr_subgoal.edges[edge]
                next = curr_edge.destiny
                distance = next.h_distance(destiny)
                next_key = str(next.x) + "," + str(next.y)
                if next_key in closed_nodes or next_key in skip:
                    continue
                if next_key in opened_nodes:
                    new_w = min(
                        curr_weight + curr_edge.weight + distance,
                        opened_nodes[next_key][0],
                    )
                    if new_w == opened_nodes[next_key][0]:
                        continue
                    opened_nodes[next_key] = (new_w, next, key)
                else:
                    opened_nodes[next_key] = (
                        curr_weight + curr_edge.weight + distance,
                        next,
                        key,
                    )

        return [], -1

    def a_grid_search(self, destiny: tuple[int, int], grid: list[list], limit: int = 0):
        key = str(self.x) + "," + str(self.y)
        opened_nodes: dict[str, tuple[int, tuple[int, int], str]] = {
            key: (0, (self.x, self.y), None)
        }
        closed_nodes: dict[str, tuple[int, tuple[int, int], str]] = {}
        while opened_nodes:
            lowest = min(opened_nodes.values(), key=lambda t: t[0])[1]
            key = str(lowest[0]) + "," + str(lowest[1])
            curr_weight, curr_subgoal, father_key = opened_nodes.pop(key)
            if curr_subgoal == destiny:
                path: list[tuple[int, int]] = []
                path.append(curr_subgoal)
                father = closed_nodes[father_key]
                while father[2] != None:
                    path.append(father[1])
                    father = closed_nodes[father[2]]
                path.append(father[1])
                path.reverse()
                return path, curr_weight
            closed_nodes[key] = (curr_weight, curr_subgoal, father_key)
            movements: list[tuple[int, int]] = [
                (1, -1),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
            ]
            for x in range(8):
                next = (
                    movements[x][0] + curr_subgoal[0],
                    movements[x][1] + curr_subgoal[1],
                )
                next_key = str(next[0]) + "," + str(next[1])
                next_weight = int(10 * ((SQRT_2 * ((x + 1) & 1)) or 1))
                distance = abs(destiny[0] - next[0]) + abs(destiny[1] - next[1])
                if (
                    next_key in closed_nodes
                    or next[0] < 0
                    or next[1] < 0
                    or next[1] >= len(grid)
                    or next[0] >= len(grid[next[1]])
                    or grid[next[1]][next[0]] == 0
                    or (
                        ((x + 1) & 1)
                        and not self.__has_diagonal(curr_subgoal, *movements[x], grid)
                    )
                ):
                    continue
                if next_key in opened_nodes:
                    new_w = min(
                        curr_weight + next_weight + distance, opened_nodes[next_key][0]
                    )
                    if new_w == opened_nodes[next_key][0]:
                        continue
                    opened_nodes[next_key] = (new_w, next, key)
                else:
                    opened_nodes[next_key] = (
                        curr_weight + next_weight + distance,
                        next,
                        key,
                    )

        return [], -1

    def h_reachable(self, destiny: "SubGoal", grid: list[list]):
        return self.h_distance(destiny) == self.a_graph_search(destiny, grid)[1]

    def h_distance(self, subgoal: "SubGoal"):
        dist_x = abs(self.x - subgoal.x)
        dist_y = abs(self.y - subgoal.y)
        return int(10 * (abs(dist_x - dist_y) + min(dist_x, dist_y) * SQRT_2))

    def add_edge(self, subgoal: "SubGoal"):
        key = str(subgoal.x) + "," + str(subgoal.y)
        if not self.edges.get(key):
            weight = self.h_distance(subgoal)
            self.edges[key] = Edge(self, subgoal, weight)
            subgoal.add_edge(self)

    def del_edge(self, x: int, y: int):
        key = str(x) + "," + str(y)
        if self.edges.get(key):
            edge = self.edges.pop(key)
            edge.destiny.del_edge(self.x, self.y)

    def __can_reduce(self, destiny: "SubGoal", distance: int):
        queue: list[tuple[int, SubGoal]] = []
        key = str(destiny.x) + "," + str(destiny.y)
        for edge in self.edges:
            if edge == key:
                continue
            next = self.edges[edge].destiny
            if self.edges[edge].weight == distance - next.h_distance(destiny):
                queue.append((self.edges[edge].weight, next))
        for curr_distance, vertice in queue:
            remaining_distance = distance - curr_distance
            if remaining_distance == 0:
                return True
            for edge in vertice.edges:
                next = vertice.edges[edge].destiny
                if vertice.edges[edge].weight == remaining_distance - next.h_distance(
                    destiny
                ):
                    queue.append((curr_distance + vertice.edges[edge].weight, next))
        return False

    def reduce_edges(self):
        keys_to_reduce = list(self.edges.keys())
        for key in keys_to_reduce:
            subgoal = self.edges[key].destiny
            edge_weight = self.edges[key].weight
            if self.__can_reduce(subgoal, edge_weight):
                self.del_edge(subgoal.x, subgoal.y)

    def __clearence(
        self,
        origin: tuple[int, int],
        dir_x: int,
        dir_y: int,
        grid: list[list],
        limit: int = -1,
        create: bool = True,
    ):
        curr = (origin[0] + dir_x, origin[1] + dir_y)
        max: int = 0
        while (
            (limit == -1 or max < limit)
            and 0 <= curr[1] < len(grid)
            and 0 <= curr[0] < len(grid[curr[1]])
        ):
            if grid[curr[1]][curr[0]] == 0:
                return max
            if type(grid[curr[1]][curr[0]]) == SubGoal:
                if create:
                    self.add_edge(grid[curr[1]][curr[0]])
                return max
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            max += 1
        return max


def create_verices(grid: list[list]):
    vertices: dict[str, SubGoal] = {}

    def check_diagonal(x: int, y: int, dir_x: int, dir_y: int, grid: list[list]):
        pos = (x + dir_x, y + dir_y)
        if (
            x < 0
            or y < 0
            or y >= len(grid)
            or x >= len(grid[y])
            or pos[0] < 0
            or pos[1] < 0
            or pos[1] >= len(grid)
            or pos[0] >= len(grid[pos[1]])
        ):
            return False
        return (
            grid[y + dir_y][x + dir_x] == 0
            and grid[y][x + dir_x] == 1 == grid[y + dir_y][x]
        )

    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value == 0:
                continue
            key = str(x) + "," + str(y)
            # Superior Esquerda
            if check_diagonal(x, y, -1, -1, grid):
                vertices[key] = SubGoal(x, y)
            # Superior Direita
            elif check_diagonal(x, y, 1, -1, grid):
                vertices[key] = SubGoal(x, y)
            # Inferior Direita
            elif check_diagonal(x, y, 1, 1, grid):
                vertices[key] = SubGoal(x, y)
            # Inferior Esquerda
            elif check_diagonal(x, y, -1, 1, grid):
                vertices[key] = SubGoal(x, y)
            else:
                continue
            grid[y][x] = vertices[key]
    return vertices


def create_grid(file_name):
    grid = [[]]
    with open(file_name, "r") as f:
        f.readline()  # figure type
        h_size = int(f.readline().strip().split(" ")[1])  # height
        f.readline()  # width
        f.readline()  # map
        grid = [[]] * h_size
        for index, row in enumerate(f.readlines()):
            grid[index] = [1 if x == "." else 0 for x in row.strip()]
    return grid


def create_ssg(file_name):
    start = time.time()
    grid = create_grid(file_name)
    # grid = [
    #     [1, 1, 1, 1, 1, 1, 0, 1],
    #     [1, 0, 0, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1],
    # ]
    print(f"{len(grid)}x{len(grid[0])} grid created in {time.time() - start} seconds")
    start = time.time()
    vertices = create_verices(grid)

    print(f"{len(vertices)} vertices created in {time.time() - start} seconds")

    start = time.time()
    for x in vertices:
        vert = vertices[x]
        vert.generate_nodes(vertices, grid)
    print(f"Nodes generated in {time.time() - start} seconds")
    start = time.time()
    for x in vertices:
        vert = vertices[x]
        vert.reduce_edges()
    print(f"Nodes reduced in {time.time() - start} seconds")
    return grid, vertices


def create_tsg(vertices: dict[str, SubGoal], grid: list[list]):
    start = time.time()
    global_goals: dict[str, SubGoal] = dict(vertices)
    local_goals: dict[str, SubGoal] = {}
    idx = 1
    for s in vertices:
        vert = vertices[s]
        new_edges: dict[str, tuple[SubGoal, SubGoal]] = {}
        local: bool = True
        visited_edges: set[str] = set()
        # print(f"{idx}/{len(vertices)}")
        idx += 1
        for p in vert.edges:
            for q in vert.edges:
                if p == q or (q, p) in visited_edges:
                    continue
                visited_edges.add((p, q))
                skip = [x for x in local_goals if x != p and x != q]
                skip.append(s)
                origin = vert.edges[p].destiny
                destiny = vert.edges[q].destiny
                _, d = origin.a_graph_search(destiny, skip)
                if d == -1 or d > vert.edges[p].weight + vert.edges[q].weight:
                    if origin.h_reachable(destiny, grid):
                        check_key = q + "," + p
                        if check_key not in new_edges:
                            new_edges[p + "," + q] = (origin, destiny)
                    else:
                        local = False
                        break
            if not local:
                break
        if local:
            for edge in new_edges:
                edge_tuple = new_edges[edge]
                edge_tuple[0].add_edge(edge_tuple[1])
            local_goals[s] = vert
    global_goals = dict(
        [(x, global_goals[x]) for x in global_goals if x not in local_goals]
    )
    print("Time to create TSG:", time.time() - start)

    return global_goals, local_goals


def main():
    grid, vertices = create_ssg("input.map")
    global_goals, local_goals = create_tsg(vertices, grid)

    start = time.time()
    plot_grid = [
        [[y * 255, y * 255, y * 255] if type(y) == int else [255, 0, 0] for y in x]
        for x in grid
    ]
    plt.imshow(plot_grid)
    total_edges = set()
    global_edges = set()
    local_edges = set()
    for x in global_goals:
        vert = global_goals[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            if (destiny.x, destiny.y, vert.x, vert.y) in total_edges:
                continue
            total_edges.add((vert.x, vert.y, destiny.x, destiny.y))
    for x in global_goals:
        vert = global_goals[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            destiny_str = str(destiny.x) + "," + str(destiny.y)
            if (
                destiny.x,
                destiny.y,
                vert.x,
                vert.y,
            ) in global_edges or destiny_str in local_goals:
                continue
            global_edges.add((vert.x, vert.y, destiny.x, destiny.y))
            plt.plot([vert.x, destiny.x], [vert.y, destiny.y], "g-", lw=0.5)
    for x in local_goals:
        vert = local_goals[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            if (destiny.x, destiny.y, vert.x, vert.y) in local_edges:
                continue
            local_edges.add((vert.x, vert.y, destiny.x, destiny.y))
            # plt.plot([vert.x, destiny.x], [vert.y, destiny.y], "g-", lw=0.5)
    print(
        f"Global edges: {len(global_edges)} = {len(total_edges)} - {len(local_edges)}"
    )
    print(f"Global Goals: {len(global_goals)} = {len(vertices)} - {len(local_goals)}")
    print(f"Plot created in {time.time() - start} seconds")
    plt.show()


if __name__ == "__main__":
    main()
