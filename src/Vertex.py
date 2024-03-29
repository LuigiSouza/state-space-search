from __future__ import annotations

NORMAL = 10
DIAGONAL = 14

Point = tuple[int, int]


class Edge:
    def __init__(
        self, origin: "Vertex", destiny: "Vertex", weight: int, is_local: bool = False
    ) -> None:
        self.origin = origin
        self.destiny = destiny
        self.weight = weight
        self.is_local = is_local


class Vertex:
    def __init__(self, x: int, y: int, grid: list[list]) -> None:
        self.x = x
        self.y = y
        self.key = str(x) + "," + str(y)
        self.edges: dict[str, Edge] = {}
        self.grid = grid

    def create_edges(self) -> None:
        """
        Finds all h-reachable vertices and adds them to the vertex's edges.
        """
        max_top = self.__clearence((self.x, self.y), 0, -1)
        max_bot = self.__clearence((self.x, self.y), 0, +1)
        max_left = self.__clearence((self.x, self.y), -1, 0)
        max_right = self.__clearence((self.x, self.y), +1, 0)
        # top right
        self.__expand_diagonal(+1, -1, max_right, max_top)
        # bottom right
        self.__expand_diagonal(+1, +1, max_right, max_bot)
        # bottom left
        self.__expand_diagonal(-1, +1, max_left, max_bot)
        # top left
        self.__expand_diagonal(-1, -1, max_left, max_top)

    def __expand_diagonal(self, dir_x: int, dir_y: int, lim_h: int, lim_v: int) -> None:
        curr = (self.x, self.y)
        while Vertex.has_diagonal(curr, dir_x, dir_y, self.grid):
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            lim_h = self.__clearence(curr, dir_x, 0, lim_h)
            lim_v = self.__clearence(curr, 0, dir_y, lim_v)
            if type(self.grid[curr[1]][curr[0]]) == Vertex:
                self.add_edge(self.grid[curr[1]][curr[0]])

    def __clearence(
        self,
        origin: Point,
        dir_x: int,
        dir_y: int,
        limit: int = -1,
        create: bool = True,
    ) -> int:
        """
        Funcion that returns how far the line parting from origin can reach before
        hitting a wall or a vertice. If hits a vertice, creates the edge between then.
        """
        curr = (origin[0] + dir_x, origin[1] + dir_y)
        max: int = 0
        while (
            (limit == -1 or max < limit)
            and 0 <= curr[1] < len(self.grid)
            and 0 <= curr[0] < len(self.grid[curr[1]])
        ):
            if self.grid[curr[1]][curr[0]] == 0:
                return max
            if type(self.grid[curr[1]][curr[0]]) == Vertex:
                if create:
                    self.add_edge(self.grid[curr[1]][curr[0]])
                return max
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            max += 1
        return max

    def h_distance(self, destiny: "Vertex") -> int:
        """
        Returns the octile distance between the vertex and destiny.
        """
        dist_x = abs(self.x - destiny.x)
        dist_y = abs(self.y - destiny.y)
        return NORMAL * max(dist_x, dist_y) + (DIAGONAL - NORMAL) * min(dist_x, dist_y)

    def add_edge(self, destiny: "Vertex") -> None:
        key = destiny.key
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
        """
        Reduces the vertex's edges if the one removed does not increase the weight
        between the two connected vertices.
        """
        keys_to_reduce = list(self.edges.keys())
        for key in keys_to_reduce:
            vertex = self.edges[key].destiny
            edge_weight = self.edges[key].weight
            if self.__can_reduce(vertex, edge_weight):
                self.del_edge(vertex.x, vertex.y)

    def __can_reduce(self, destiny: "Vertex", distance: int) -> bool:
        """
        Checks if the weight between two vertices is the same if traveled between
        its neighbors.
        """
        queue: list[tuple[int, Vertex]] = []
        key = destiny.key
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

    def is_out_of_bounds(x: int, y: int, grid: list[list]) -> bool:
        """
        Funcion to check if given coordinates are outside the grid.
        """
        return x < 0 or y < 0 or y >= len(grid) or x >= len(grid[y])

    def has_diagonal(
        origin: Point,
        dir_x: int,
        dir_y: int,
        grid: list[list],
    ) -> bool:
        """
        Function to check if a given point has access to its diagonal given it
        his direction.
        """
        x = origin[0]
        y = origin[1]
        target = (x + dir_x, y + dir_y)

        if Vertex.is_out_of_bounds(x, y, grid) or Vertex.is_out_of_bounds(
            target[0], target[1], grid
        ):
            return False
        if (
            grid[target[1]][target[0]] == 0
            or grid[target[1]][x] == 0
            or grid[y][target[0]] == 0
        ):
            return False
        return True
