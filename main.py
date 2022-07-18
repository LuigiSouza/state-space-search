import queue
import matplotlib.pyplot as plt
import time

SQRT_2 = 14


class Edge:
    def __init__(self, origin: "Vertex", destiny: "Vertex", weight: int):
        self.origin = origin
        self.destiny = destiny
        self.weight = weight


class Vertex:
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
        curr = (self.x, self.y)
        local_hor_limit = max_right
        local_ver_limit = max_top
        while self.__has_diagonal(curr, 1, -1, grid):
            curr = (curr[0] + 1, curr[1] - 1)
            local_hor_limit = self.__clearence(curr, 1, 0, grid, local_hor_limit)
            local_ver_limit = self.__clearence(curr, 0, -1, grid, local_ver_limit)
            if type(grid[curr[1]][curr[0]]) == Vertex:
                self.add_edge(grid[curr[1]][curr[0]])
        # bottom right
        curr = (self.x, self.y)
        local_hor_limit = max_right
        local_ver_limit = max_bot
        while self.__has_diagonal(curr, 1, 1, grid):
            curr = (curr[0] + 1, curr[1] + 1)
            local_hor_limit = self.__clearence(curr, 1, 0, grid, local_hor_limit)
            local_ver_limit = self.__clearence(curr, 0, 1, grid, local_ver_limit)
            if type(grid[curr[1]][curr[0]]) == Vertex:
                self.add_edge(grid[curr[1]][curr[0]])
        # bottom left
        curr = (self.x, self.y)
        local_hor_limit = max_left
        local_ver_limit = max_bot
        while self.__has_diagonal(curr, -1, 1, grid):
            curr = (curr[0] - 1, curr[1] + 1)
            local_hor_limit = self.__clearence(curr, -1, 0, grid, local_hor_limit)
            local_ver_limit = self.__clearence(curr, 0, 1, grid, local_ver_limit)
            if type(grid[curr[1]][curr[0]]) == Vertex:
                self.add_edge(grid[curr[1]][curr[0]])
        # top left
        curr = (self.x, self.y)
        local_hor_limit = max_left
        local_ver_limit = max_top
        while self.__has_diagonal(curr, -1, -1, grid):
            curr = (curr[0] - 1, curr[1] - 1)
            local_hor_limit = self.__clearence(curr, -1, 0, grid, local_hor_limit)
            local_ver_limit = self.__clearence(curr, 0, -1, grid, local_ver_limit)
            if type(grid[curr[1]][curr[0]]) == Vertex:
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

    def distance(self, vertex: "Vertex"):
        dist_x = abs(self.x - vertex.x)
        dist_y = abs(self.y - vertex.y)
        return 10 * (abs(dist_x - dist_y) + min(dist_x, dist_y) * SQRT_2)

    def add_edge(self, vertex: "Vertex"):
        key = str(vertex.x) + "," + str(vertex.y)
        if not self.edges.get(key):
            weight = self.distance(vertex)
            self.edges[key] = Edge(self, vertex, weight)
            vertex.add_edge(self)

    def del_edge(self, x: int, y: int):
        key = str(x) + "," + str(y)
        if self.edges.get(key):
            edge = self.edges.pop(key)
            edge.destiny.del_edge(self.x, self.y)

    def __can_reduce(self, destiny: "Vertex", distance: int):
        print("Reducing")
        queue: list[tuple[int, Vertex]] = []
        key = str(destiny.x) + "," + str(destiny.y)
        for edge in self.edges:
            if edge == key:
                continue
            next = self.edges[edge].destiny
            if self.distance(next) == distance - next.distance(destiny):
                queue.append((self.distance(next), next))
        for new_distance, vertice in queue:
            print("lista", queue)
            new_distance = distance - new_distance
            if new_distance == 0:
                return True
            for edge in vertice.edges:
                next = vertice.edges[edge].destiny
                if vertice.distance(next) == new_distance - next.distance(destiny):
                    queue.append((vertice.distance(next), next))
        return False

    def reduce_edges(self):
        keys_to_reduce = list(self.edges.keys())
        for key in keys_to_reduce:
            vertex = self.edges[key].destiny
            edge_weight = self.edges[key].weight
            if self.__can_reduce(vertex, edge_weight):
                self.del_edge(vertex.x, vertex.y)

    def __clearence(
        self,
        origin: tuple[int, int],
        dir_x: int,
        dir_y: int,
        grid: list[list],
        limit: int = -1,
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
            if type(grid[curr[1]][curr[0]]) == Vertex:
                self.add_edge(grid[curr[1]][curr[0]])
                return max
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            max += 1
        return max


def create_verices(grid):
    vertices: dict[str, Vertex] = {}
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value == 0:
                continue
            key = str(x) + "," + str(y)
            # Superior Esquerda
            if (
                y > 0 < x
                and grid[y - 1][x - 1] == 0
                and grid[y][x - 1] == 1 == grid[y - 1][x]
            ):
                vertices[key] = Vertex(x, y)
            # Superior Direita
            elif (
                x > 0
                and y < len(grid) - 1
                and grid[y + 1][x - 1] == 0
                and grid[y][x - 1] == 1 == grid[y + 1][x]
            ):
                vertices[key] = Vertex(x, y)
            # Inferior Direita
            elif (
                x < len(row) - 1
                and y < len(grid) - 1
                and grid[y + 1][x + 1] == 0
                and grid[y][x + 1] == 1 == grid[y + 1][x]
            ):
                vertices[key] = Vertex(x, y)
            # Inferior Esquerda
            elif (
                x < len(row) - 1
                and y > 0
                and grid[y - 1][x + 1] == 0
                and grid[y][x + 1] == 1 == grid[y - 1][x]
            ):
                vertices[key] = Vertex(x, y)
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


def main():
    start = time.time()
    # grid = create_grid("input.map")
    grid = [
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    print(f"Grid created in {time.time() - start} seconds")
    start = time.time()
    vertices = create_verices(grid)

    print(f"Vertices created in {time.time() - start} seconds")

    start = time.time()
    plot_grid = [
        [[y * 255, y * 255, y * 255] if type(y) == int else [255, 0, 0] for y in x]
        for x in grid
    ]
    for x in vertices:
        vert = vertices[x]
        vert.generate_nodes(vertices, grid)
    for x in vertices:
        vert = vertices[x]
        vert.reduce_edges()
    print(f"Nodes generated in {time.time() - start} seconds")
    start = time.time()
    plt.imshow(plot_grid)
    edges = set()
    for x in vertices:
        vert = vertices[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            if (destiny.x, destiny.y, vert.x, vert.y) in edges:
                continue
            edges.add((vert.x, vert.y, destiny.x, destiny.y))
            plt.plot([vert.x, destiny.x], [vert.y, destiny.y], "g-", lw=0.5)
    print("Total edges:", len(edges))
    print(f"Plot created in {time.time() - start} seconds")
    plt.show()


if __name__ == "__main__":
    main()
