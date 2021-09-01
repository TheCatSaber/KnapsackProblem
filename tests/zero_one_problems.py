"""zero_one_problems.py: 0-1 problems for tests."""
# Copyright (C) 2021 TheCatSaber

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import NamedTuple

from context import KnapsackProblem, KnapsackSolution

# Create named tuple for problem, answer

ZOp = NamedTuple("ZOp", (("p", KnapsackProblem), ("a", KnapsackSolution)))

ZOpBaseCase = ZOp(p=KnapsackProblem(0, [], [], 0), a=(0, []))

ZOpSimple = ZOp(p=KnapsackProblem(1, [1], [5], 2), a=(5, [1]))

ZOpWP = ZOp(
    p=KnapsackProblem(5, [1, 1, 2, 4, 12], [1, 2, 2, 10, 4], 15),
    a=(15, [1, 1, 1, 1, 0]),
)

# These tests are based on
# https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html

ZOp01 = ZOp(
    p=KnapsackProblem(
        10,
        [23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
        [92, 57, 49, 68, 60, 43, 67, 84, 87, 72],
        165,
    ),
    a=(309, [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
)

ZOp02 = ZOp(
    p=KnapsackProblem(5, [12, 7, 11, 8, 9], [24, 13, 23, 15, 16], 26),
    a=(51, [0, 1, 1, 1, 0]),
)

ZOp03 = ZOp(
    p=KnapsackProblem(6, [56, 59, 80, 64, 75, 17], [50, 50, 64, 46, 50, 5], 190),
    a=(150, [1, 1, 0, 0, 1, 0]),
)

ZOp04 = ZOp(
    p=KnapsackProblem(7, [31, 10, 20, 19, 4, 3, 6], [70, 20, 39, 37, 7, 5, 10], 50),
    a=(107, [1, 0, 0, 1, 0, 0, 0]),
)

ZOp05 = ZOp(
    p=KnapsackProblem(
        8, [25, 35, 45, 5, 25, 3, 2, 2], [350, 400, 450, 20, 70, 8, 5, 5], 104
    ),
    a=(900, [1, 0, 1, 1, 1, 0, 1, 1]),
)

ZOp06 = ZOp(
    p=KnapsackProblem(
        7,
        [41, 50, 49, 59, 55, 57, 60],
        [442, 525, 511, 593, 546, 564, 617],
        170,
    ),
    a=(1735, [0, 1, 0, 1, 0, 0, 1]),
)

ZOp07 = ZOp(
    p=KnapsackProblem(
        15,
        [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120],
        [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240],
        750,
    ),
    a=(1458, [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]),
)

ZOp08 = ZOp(
    p=KnapsackProblem(
        24,
        [
            382745,
            799601,
            909247,
            729069,
            467902,
            44328,
            34610,
            698150,
            823460,
            903959,
            853665,
            551830,
            610856,
            670702,
            488960,
            951111,
            323046,
            446298,
            931161,
            31385,
            496951,
            264724,
            224916,
            169684,
        ],
        [
            825594,
            1677009,
            1676628,
            1523970,
            943972,
            97426,
            69666,
            1296457,
            1679693,
            1902996,
            1844992,
            1049289,
            1252836,
            1319836,
            953277,
            2067538,
            675367,
            853655,
            1826027,
            65731,
            901489,
            577243,
            466257,
            369261,
        ],
        6404180,
    ),
    a=(
        (
            13549094,
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        )
    ),
)
