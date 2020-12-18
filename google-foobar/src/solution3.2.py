class FindExitPath():
    width, height = 0, 0
    shortest_path = 3  # min value for 2X2 array
    mini_path = 1000
    maze = None

    def solve_maze(self, maze, height, width):

        self.height = height
        self.width = width
        self.maze = maze
        self.shortest_path = height + width - 1
        self.mini_path = 1000  # max value

        # Iterate over all possible inputs (by replacing 1s with 0s).
        for maze_0 in self.all_possible_maps(self.maze):

            # Find the minimal path length
            self.mini_path = min(self.get_min_path(maze_0, width, height), self.mini_path)

            # min path is the optimal then return min
            if self.mini_path == self.shortest_path:
                return self.mini_path

        return self.mini_path

    def get_min_path(self, maze_i, w, h):
        '''Takes a map m and returns the minimal path
        from the start to the end node. Also pass width and height.
        '''
        path_dict = {1: {(0, 0)}}

        # Expand "fringe" successively
        path_length = 2
        while path_length < 1000 and path_dict[path_length - 1]:

            # Fill fringe
            fringe = set()
            for x in path_dict[path_length - 1]:
                # Expand node x (all neighbors) and exclude already visited
                expand_x = {y for y in self.get_neighbors(x, maze_i) if
                            not any(y in visited for visited in path_dict.values())}
                fringe = fringe | expand_x

            # Have we found min path of exit node?
            if (h - 1, w - 1) in fringe:
                return path_length

            # Store new fring of minimal-path nodes
            path_dict[path_length] = fringe

            # Find nodes with next-higher path_length
            path_length += 1

        return 1000  # Infinite path length

    def all_possible_maps(self, maze):
        '''Returns an iterator for memory efficiency
        over all maps that arise by replacing a '1' with a '0' value.'''

        yield maze

        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j]:
                    # Copy the map
                    copy = [[col for col in row] for row in maze]

                    # Replace 1 by 0 and yield new map
                    copy[i][j] = 0
                    yield copy

    def get_neighbors(self, x, maze_i):
        '''
        Returns a set of nodes (as tuples) that are neighbors of node x in given maze.
        '''
        i, j = x
        w, h = len(maze_i[0]), len(maze_i)
        possible_moves = {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
        neighbors_set = set()
        for y in possible_moves:
            i, j = y
            if i >= 0 and i < h and j >= 0 and j < w and maze_i[i][j] == 0:
                neighbors_set.add(y)

        return neighbors_set


def solution(map):
    width = len(map[0])
    height = len(map)
    fp = FindExitPath()
    return fp.solve_maze(map, height, width)


maze1 = [[0, 1, 1], [0, 0, 0], [1, 1, 0]]  # Answer = 5
maze2 = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]  # Answer =7
maze3 = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0]]  # Answer =11
maze4 = [[0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1],
         [0, 0, 0, 0, 0]]  # Answer =10
maze5 = [[0, 1, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0]]  # Answer =11
maze6 = [[0, 1, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 1, 1, 0],
         [0, 0, 1, 1, 0]]  # Answer =11
maze7 = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]  # Answer 7

usecase_list = [maze1, maze2, maze3, maze4, maze5, maze6, maze7]

i = 1
for case in usecase_list:
    print("---------------------------------------------")
    print(f"User case :  {i}")
    print("---------------------------------------------")
    print(f" Result : {solution(case)}")
    print("=============================================")

    i += 1
