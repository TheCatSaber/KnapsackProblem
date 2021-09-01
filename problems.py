"""problems.py: Class to store a Knapsack Problem."""
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


class KnapsackProblem:
    """Class to store a Knapsack Problem."""

    @classmethod
    def verify(cls, n: int, w: list[int], v: list[int], W: int) -> None:
        """Verify Knapsack Problem is valid."""
        if n < 0:
            raise ValueError(
                "The number of items for the knapsack problem must be greater than or"
                " equal to 0."
            )
        if len(w) != n:
            raise ValueError("The length of the weights list must be equal to n.")
        if len(v) != n:
            raise ValueError("The length of the values list must be equal to n.")
        if any(i < 0 for i in w):
            raise ValueError(
                "The weights of the items must all be greater than or equal to 0."
            )
        if any(i < 0 for i in v):
            raise ValueError(
                "The values of the items must all be greater than or equal to 0."
            )
        if W < 0:
            raise ValueError(
                "The maximum weight (W) must be greater than or equal to 0."
            )

    def __init__(self, n: int, w: list[int], v: list[int], W: int) -> None:
        """Class to store a Knapsack Problem."""
        # verify raises errors and no return value, so just call it
        self.verify(n, w, v, W)
        self.n = n
        self.w = w
        self.v = v
        self.W = W
        # x is currently not used, but may be used in the future by unbounded/bounded problems
        # and if it is, then current programs should be adapted to use it.
        self.x = [0] * n  # Init list counting copies of each item to be used
