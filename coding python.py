
list.append(var)
list.pop()

list.insert(ind, var)
list.pop(ind)

squared = list(map(lambda x: x**2, items))
less_than_zero = list(filter(lambda x: x < 0, items))
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])

sorted(list, key=lambda x: x>0)
reversed(list)

def cmp(a, b):
    return -1 if a < b else (1 if a > b else 0)

from functools import cmp_to_key
sorted(mylist, key=cmp_to_key(cmp))

for counter, value in enumerate(some_list):

from copy import copy, deepcopy

frozenset

all, any

from heapq import heappush, heappop, heapify
heappush(heap, item)
heappop(heap)
heap[0] # peek

yield

*:
 multiply sequences
 function declaration: convert varialble number of arguments into sequence (** convert named variables into dictionary)
 in function call: unpack sequence into positional arguments (** unpack dictionaries into named arguments)

collections.deque
append, appendleft, pop, popleft

collections.Counter
most_common

collections.defaultdict
d = defaultdict(list) # alternatively get(val, default_val)

math.inf, math.nan
sys.maxsize

import datetime as dt

datetime.datetime(2019, 10, 11, 11, 15, 48, 164461)
delta = dt.timedelta(days=100)

compare objects:
__lt__, __eq__


