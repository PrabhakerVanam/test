import string
def solution(x):
    asci_l = string.ascii_letters[0:26]
    res = ''
    for letter in x:
        if letter in string.ascii_lowercase:
            res = res + ''.join(sorted(asci_l, reverse=True)[asci_l.find(letter)])
        else:
            res = res + letter
    return res

print(solution("vmxibkgrlm"))