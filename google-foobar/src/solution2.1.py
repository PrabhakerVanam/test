from itertools import combinations


def tuple_to_num(tups):
    return int(''.join(map(str, tups)))


def solution(L):
    L.sort(reverse=True)
    res = 0
    if sum(L) % 3 != 0:
        # print(L)
        for idx in reversed(range(1, len(L) + 1)):
            # print(idx)
            for tups in combinations(L, idx):
                # print(tups)
                if sum(tups) % 3 == 0:
                    # print("its divisible", sum(tups))
                    return tuple_to_num(tups)

    elif sum(L) != 0 and sum(L) % 3 == 0:
        return tuple_to_num(tuple(L))

    return res


list1 = [3, 1, 4, 1, 5, 9]
list2 = [3, 1, 4, 1]
list3 = [1, 6, 9, 3, 4, 5, 7, 5, 1, 4]

print(solution(list1))

print(solution(list2))

print(solution(list3))

print(solution([2, 9, 2]))

print(solution([0, 0, 0]))
