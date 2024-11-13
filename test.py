import random

l = [1, 2, 3, 4]
ws = [1, 1, 1, 3]

picks = random.choices(l, ws, k=100000)
for el in l:
    print(picks.count(el) / 100000)
