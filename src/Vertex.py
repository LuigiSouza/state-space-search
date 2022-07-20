SQRT_2 = 1.4


class Edge:
    def __init__(self, origin: "Vertex", destiny: "Vertex", weight: int) -> None:
        self.origin = origin
        self.destiny = destiny
        self.weight = weight


class Vertex:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.edges: dict[str, Edge] = {}

    def create_edges(self, grid: list[list]) -> None:
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

    def h_distance(self, destiny: "Vertex") -> int:
        dist_x = abs(self.x - destiny.x)
        dist_y = abs(self.y - destiny.y)
        return int(10 * (abs(dist_x - dist_y) + min(dist_x, dist_y) * SQRT_2))

    def add_edge(self, destiny: "Vertex") -> None:
        key = str(destiny.x) + "," + str(destiny.y)
        if not self.edges.get(key):
            weight = self.h_distance(destiny)
            self.edges[key] = Edge(self, destiny, weight)
            destiny.add_edge(self)

    def del_edge(self, x: int, y: int) -> None:
        key = str(x) + "," + str(y)
        if self.edges.get(key):
            edge = self.edges.pop(key)
            edge.destiny.del_edge(self.x, self.y)

    def reduce_edges(self) -> None:
        keys_to_reduce = list(self.edges.keys())
        for key in keys_to_reduce:
            vertex = self.edges[key].destiny
            edge_weight = self.edges[key].weight
            if self.__can_reduce(vertex, edge_weight):
                self.del_edge(vertex.x, vertex.y)

    def __expand_diagonal(
        self, dir_x: int, dir_y: int, lim_h: int, lim_v: int, grid: list[list]
    ) -> None:
        curr = (self.x, self.y)
        while Vertex.has_diagonal(curr, dir_x, dir_y, grid):
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            lim_h = self.__clearence(curr, dir_x, 0, grid, lim_h)
            lim_v = self.__clearence(curr, 0, dir_y, grid, lim_v)
            if type(grid[curr[1]][curr[0]]) == Vertex:
                self.add_edge(grid[curr[1]][curr[0]])

    def __can_reduce(self, destiny: "Vertex", distance: int) -> bool:
        queue: list[tuple[int, Vertex]] = []
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

    def has_diagonal(
        origin: tuple[int, int],
        dir_x: int,
        dir_y: int,
        grid: list[list],
    ) -> bool:
        x = origin[0]
        y = origin[1]
        target = (x + dir_x, y + dir_y)

        if (
            x < 0
            or y < 0
            or x >= len(grid)
            or y >= len(grid[0])
            or target[0] < 0
            or target[1] < 0
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

    def __clearence(
        self,
        origin: tuple[int, int],
        dir_x: int,
        dir_y: int,
        grid: list[list],
        limit: int = -1,
        create: bool = True,
    ) -> int:
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
                if create:
                    self.add_edge(grid[curr[1]][curr[0]])
                return max
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            max += 1
        return max
