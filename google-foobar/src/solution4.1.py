
def solution(banana_list):
    banana_list = sorted(banana_list, reverse=False)
    print(banana_list)

    guards = [[] for i in range(len(banana_list))]

    non_loop_guards = 0

    for i in range(len(guards)):
        for j in range(len(guards)):
            if is_loop_pair(banana_list[i], banana_list[j]):
                guards[i].append(j)

    print(guards)
    guards_count = len(banana_list)
    while guards_count > 0:
        
        cur_idx = 0
        for i in range(len(guards)):
            
            if (i != 0 and (len(guards[i]) < len(guards[cur_idx]) or guards[cur_idx]
                            == [-1]) and guards[i] != [-1]):
                cur_idx = i
            print(f"cur_idx - {cur_idx}")
        if ((len(guards[cur_idx])) == 0 or (len(guards[cur_idx]) == 1 and
                                            guards[cur_idx][0] == guards[cur_idx]) and guards[cur_idx] != [-1]):
            remove(guards, cur_idx)
            guards_count -= 1
            non_loop_guards += 1
            print(f"guards_count - {guards_count}")
            print(f"non_loop_guards - {non_loop_guards}")

        else:
            min_node = guards[cur_idx][0]
            print(min_node)
            for i in range(len(guards[cur_idx])):
                if (i != 0 and guards[cur_idx][i] != cur_idx and len(guards[guards[cur_idx][i]]) < len(guards[min_node])):
                    min_node = guards[cur_idx][i]
                    print(min_node)
            if guards[min_node] != [-1]:
                remove(guards, cur_idx)
                remove(guards, min_node)
                guards_count -= 2

    print(non_loop_guards)

    return non_loop_guards


def is_loop_pair(a, b):
    
    total = a + b
    redu_total = total
    while redu_total % 2 == 0:
        redu_total = redu_total / 2
            
    return bool((a % redu_total) != 0)


def remove(guards, cur_idx):
    for i in range(len(guards)):
        j = 0
        while j < len(guards[i]):
            if (guards[i][j] == cur_idx):
                guards[i].pop(j)
            j += 1
    guards[cur_idx] = [-1]



print("==============================================")
print(solution([1, 1])) #2

print("==============================================")
print(solution([1, 7, 3, 21, 13, 19])) #0

print("==============================================")
print(solution([1, 2, 1, 7, 3, 21, 13, 19]))  #0

print("==============================================")
print(solution([1]))  #1

print("==============================================")
print(solution([1, 7, 1, 1]))  #4

print("==============================================")