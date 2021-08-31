# KnapsackProblem

Various ways to solve knapsack problems, taken from different sources.

Check [License](#license) for information of reuse.

Many of the tests are based on John Burkardt's dataset
[KNAPSACK_01](https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html).

# Recursive Explanation

What follows in an explanation of the principle type of recursion used in the project,
since this is repetitive to explain multiple times,
to be linked to from the solvers themselves (welcome if you came from `solvers.py`).

This explanation assumes 1D list indexes start at 1 (for simplicity), which is why in the code,
1D list indexes will be 1 less than the values in this explanation.

`p(i, j)` is the value of the partial solution using
`i` items and a maximum weight of `j`, such that `p(n, W)` will solve the knapsack problem.

`p(i, j)` might be a function or a 2D matrix (e.g. `m[i][j]`) (and it might have tail
arguments, such as the `knapsackProblem` object, or the matrix object.)

The following assumes that each weight is > 0 and that W is greater than 0.

`p(i, j)` can be defined recursively:

## 1. `p(0, j)` and `p(i, 0)` are 0

(No items or no weight: no value)

Explanation:
- If `i` is 0 you have no items,
and with no items you cannot have any value.
- If `j` is 0, you have no weight capacity so cannot carry items,
so cannot have any value.

## 2. If `w[i] > j` then `p(i, j) = p(i - 1, j)`

(Can item possibly fit in knapsack?)

Explanation:
- If the weight of the `i`th item is greater than `j`, then
the `i`th item cannot be in this partial solution, as it will not fit.
- Therefore, use the partial solution with one less item
(i.e. excluding the `i`th item) and the same weight.

## 3. Else, `p(i, j) = max(p(i - 1, j), p(i - 1, j - w[i]) + v[i])`

(Choose highest value of including item and not including item)

Explanation:
- The first of the values in the `max()` is the value of the partial solution without
the `i`th item, and the same weight.
- The second of the values in the `max()` is the value of the partial solution without
the `i`th item, without the weight of the `i`th item plus the value of the `i`th item.
- The first value is the value without the `i`th item, the second value is the value
with the `i`th item.
- The `max()` choses whichever of these is bigger to use as `p(i, j)`

## Solution items

To find the items in the solution, we can retrace the steps taken during the
evaluation of `p(n, W)`, by seeing if the value increases when item `i` is introduced.

If an item is introduced and the value increases, then this item is in the solution.

If an item is introduced and the value stays the same, it is not in the solution.

In `solvers.py`, the recursive solution functions (`s(i, j)`) return a list (`x`),
which is always length `n`. If `x[i]` is 0, then item
`i` is not in the solution. If `x[i]` is 1, then item `i` is in the solution.

The way this is evaluated recursively (so whether other values are in the solution is know) is:

1. Base case: if `i` is 0, return list of `n` 0s.
2. If `p(i, j)` > `p(i - 1, j)` then we call `s(i - 1, j - w[i])`,
and then set `x[i]` in this to be 1. `x` is then returned.
3. Otherwise, `s(i-1, j)` is returned.


Explanation:
- If there are no items yet, then there should be no items in the solution yet.
- If an item is introduced, `s` should be called without this item and without
the weight of this item (like for `p`).
- Otherwise, `s` should be called without this item (also like for `p`).

# license

The code ("This program") is available under 
[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html).

A copy is included
[here](/LICENSE).
 
Copyright (C) 2020 TheCatSaber

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.