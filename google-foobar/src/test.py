def solution(m):
    # Calculate map stats
    w, h = len(m[0]), len(m)  # width and height

    # The current minimal solution
    s_min = 401

    # Iterate over all possible inputs (by replacing 1s with 0s).
    for m_0 in all_maps(m):
        print(m_0)
        # Find the minimal path length
        s_min = min(min_path(m_0, w, h), s_min)

        # Optimization: Minimal solution?
        if s_min == w + h - 1:
            return s_min

    return s_min


def min_path(m, w, h):
    '''Takes a map m and returns the minimal path
    from the start to the end node. Also pass width and height.
    '''

    # Initialize dictionary of path lengths
    # integer: {(i,j), ...} (Set of nodes (i,j) with this integer path length)
    d = {1: {(0, 0)}}

    # Expand "fringe" successively
    path_length = 2
    while path_length < 401 and d[path_length - 1]:

        # Fill fringe
        fringe = set()
        for x in d[path_length - 1]:
            # Expand node x (all neighbors) and exclude already visited
            expand_x = {y for y in neighbors(x, m) if not any(y in visited for visited in d.values())}
            print(f"expand_x - {expand_x}")
            fringe = fringe | expand_x
            print(f"fringe - {fringe}")
            print(f"d - {d}")

        # Have we found min path of exit node?
        if (h - 1, w - 1) in fringe:
            return path_length

        # Store new fring of minimal-path nodes
        d[path_length] = fringe

        # Find nodes with next-higher path_length
        path_length += 1

    return 401  # Infinite path length


def neighbors(x, m):
    '''Returns a set of nodes (as tuples) that are neighbors of node x in m.'''
    i, j = x
    w, h = len(m[0]), len(m)
    candidates = {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
    neighbors = set()
    for y in candidates:
        i, j = y
        if i >= 0 and i < h and j >= 0 and j < w and m[i][j] == 0:
            neighbors.add(y)

    return neighbors


def all_maps(m):
    '''Returns an iterator for memory efficiency
    over all maps that arise by replacing a '1' with a '0' value.'''

    # Unchanged map is a valid solution
    yield m
    print(m)
    for i in range(len(m)):
        print(f"i: {i}")
        for j in range(len(m[i])):
            print(f" j: {j}")
            if m[i][j]:
                print(f"({i},{j}) - {m[i][j]}")
                # Copy the map
                copy = [[col for col in row] for row in m]
                print(copy)

                # Replace 1 by 0 and yield new map
                copy[i][j] = 0
                print(copy)
                yield copy


maze1 = [[0, 1, 1], [0, 0, 0], [1, 1, 0]]  # Answer = 5
maze2 = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]  # Answer =7
maze3 = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]  # Answer =11
maze4 = [[0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]]  # Answer =10
maze5 = [[0, 1, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0]]  # Answer =11
maze6 = [[0, 1, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 0]]  # Answer =11
maze7 = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]  # Answer 7

usecase_list = [maze1, maze2, maze3, maze4, maze5, maze6, maze7]

i = 1
for case in usecase_list:
    print("---------------------------------------------")
    print(f"User case :  {i}")
    print("---------------------------------------------")
    print(f" Result : { solution(case)}")
    print("=============================================")

    i +=1