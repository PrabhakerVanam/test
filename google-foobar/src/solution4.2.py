
def solution(entrances, exits, path):
    ent_len = len(entrances)
    pat_len = len(path)
    exi_len = len(exits)
    bunnies_count = 0
    intermediate_paths = path[ent_len:(pat_len-exi_len)]
    for i in range(pat_len - ent_len - exi_len):
        sum_inter = sum(intermediate_paths[i])
        sum_enter = 0
        for j in entrances:
            sum_enter += path[j][ent_len + i]
        bunnies_count += min(sum_enter, sum_inter)
    return bunnies_count


print(solution([0], [3], [[0, 7, 0, 0], [0, 0, 6, 0], [0, 0, 0, 8], [9, 0, 0, 0]]))
#Output: 6

print(solution([0, 1], [4, 5], [[0, 0, 4, 6, 0, 0], [0, 0, 5, 2, 0, 0], [0, 0, 0, 0, 4, 4], [0, 0, 0, 0, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))
#Output: 16