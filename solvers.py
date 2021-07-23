"""solvers.py: Classes to solve Knapsack Problems."""
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


from abc import ABC, abstractmethod
from functools import lru_cache
import itertools

from problems import KnapsackProblem

KnapsackSolution = tuple[int, list[int]]


class KnapsackSolver(ABC):
    """Abstract Base Class for Methods to solve Knapsack Problems
    defined by problem_class.KnapsackProblem objects"""

    @abstractmethod
    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Abstract classmethod to sovle kp,
        returning maximum value and list of indices of items in optimal knapsack"""

    # Things used by multiple sub-algorithms.
    @classmethod
    def check_strictly_positive(cls, w: list[int], W: int, error_name: str) -> None:
        """Checks that all weights in w and W are strcitly positive (>0).
        Raises ValueError if not."""
        if any(i <= 0 for i in w):
            raise ValueError(f"{error_name} requires all weights to be greater than 0.")
        if W <= 0:
            raise ValueError(f"{error_name} requires max weights to be greater than 0.")

    @classmethod
    def _recursive_index_dp(
        cls, i: int, j: int, m: list[list[int]], kp: KnapsackProblem
    ) -> list[int]:
        """Recursively get solution:
        if an item is introduced and value changes, that item is in the solution."""
        # Base case
        if i == 0:
            return []
        # Check for value being -1 (since certain algs set values to be -1 as default)
        # If the array value is on the edge of the array (i == 0 or j <= 0),
        # it should be 0, so set it to this.
        # Otherwise raise ValueError, as unsure what it should be.
        # i == 0 technically redudant, but kept for consistency
        if m[i][j] == -1 and (i == 0 or j <= 0):
            m[i][j] = 0
        if m[i - 1][j] == -1 and (i - 1 == 0 or j <= 0):
            m[i - 1][j] = 0
        if m[i][j] == -1 or m[i - 1][j] == -1:
            raise ValueError(
                "Not set value in m that is not on the top row or left column of"
                " the grid",
                i,
                j,
                m[i][j],
                m[i - 1][j],
            )
        # If item in solution, call recursively, add item to list and return
        if m[i][j] > m[i - 1][j]:
            old_list = cls._recursive_index_dp(i - 1, j - kp.w[i - 1], m, kp)
            old_list.append(i - 1)  # i - 1 so index of item
            return old_list
        # Otherwise item is not in, and remove one item
        else:
            return cls._recursive_index_dp(i - 1, j, m, kp)

    @classmethod
    def get_dp_solution_indexes(
        cls, kp: KnapsackProblem, m: list[list[int]]
    ) -> list[int]:
        """Get indexes of items that are in the solution.

        kp - KnapsackProblem.

        m - 2D array of values (not nesc. all filled in),
        as explained by docstrings of ZeroOneDynamicProgrammingSolver
        and ZeroOneDynamicProgrammingSolverSlow."""

        return cls._recursive_index_dp(kp.n, kp.W, m, kp)


class ZeroOneDynamicProgrammingSolver(KnapsackSolver):
    """Solve 0-1 Knapsack Problem, using Dynamic Programming,
    only calculating values needed (recursively),
    using algorithm of https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem
    """

    def _recursive(self, i: int, j: int, m: list[list[int]], kp: KnapsackProblem) -> int:
        if i == 0 or j <= 0:
            m[i][j] = 0
            return 0
        # m[i - 1][j] not calculated, so calculate it
        if m[i - 1][j] == -1:
            m[i - 1][j] = self._recursive(i - 1, j, m, kp)
        # item cannot fit in recursive_name_dp
        if kp.w[i - 1] > j:
            m[i][j] = m[i - 1][j]
        else:
            # m[i - 1][j - kp.w[i - 1]] not calculated, so caculate it
            if m[i - 1][j - kp.w[i - 1]] == -1:
                m[i - 1][j - kp.w[i - 1]] = self._recursive(
                    i - 1, j - kp.w[i - 1], m, kp
                )
            m[i][j] = max(m[i - 1][j], m[i - 1][j - kp.w[i - 1]] + kp.v[i - 1])
        return m[i][j]

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by Dynamic Programming.
        Recursively calculate values needed in m, starting at m[n][W].

        m = 2D array, with first index (i) indicating how many indexes in kp.w and kp.v to use
        and the second index (w) indicating the maximum weight for the sub-problem.
        The value of the array is the maximum value that an be attained with these constraints.

        So the solver wants to calculate m[n][W]
        m[i][w] can be defined recursively:
        • m[0][w] = 0
        • m[i][w] = m[i - 1, w] if w[i-1] > w
            ->  new item is more than current weight limit, so don't add this item,
                so value stays the same
        • m[i][w] = max(m[i-1][w], m[i-1][w-w[i-1]] + v[i-1]) if wi <= w
            ->  new item is less than current weight limit, so pick greater value of:
                • Not using new item
                • Using new item - value of (current weight limit - weight of new item) + value of new item
        """
        # Base case
        if kp.n == 0:
            return 0, []

        # This algorithm assumes w1, w2, ... wn, W > 0, so test this
        self.check_strictly_positive(kp.w, kp.W, "ZeroOneDynamicProgrammingSolver")

        m = [[-1 for _ in range(kp.W + 1)] for _ in range(kp.n + 1)]

        return self._recursive(kp.n, kp.W, m, kp), self.get_dp_solution_indexes(kp, m)


class ZeroOneDynamicProgrammingSolverSlow(KnapsackSolver):
    """Solve 0-1 Knapsack Problem, using dynamic programming,
    calculating all values needed in array,
    using algorithm of https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem
    """

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by Dynamic Programming.
        Calculates all values needed in array.

        m = 2D array, with first index (i) indicating how many indexes in kp.w and kp.v to use
        and the second index (w) indicating the maximum weight for the sub-problem.
        The value of the array is the maximum value that an be attained with these constraints.

        So the solver wants to calculate m[n][W]
        m[i][w] can be defined recursively:
        • m[0][w] = 0
        • m[i][w] = m[i - 1, w] if w[i-1] > w
            ->  new item is more than current weight limit, so don't add this item,
                so value stays the same
        • m[i][w] = max(m[i-1][w], m[i-1][w-w[i-1]] + v[i-1]) if wi <= w
            ->  new item is less than current weight limit, so pick greater value of:
                • Not using new item
                • Using new item - value of (current weight limit - weight of new item) + value of new item
        """
        if kp.n == 0:
            return 0, []

        # This algorithm assumes w1, w2, ... wn, W > 0, so test this
        self.check_strictly_positive(kp.w, kp.W, "ZeroOneDynamicProgrammingSolverSlow")

        # Creating with 0s means do not have to deal with top row and left column manually
        m = [[0 for _ in range(kp.W + 1)] for _ in range(kp.n + 1)]

        for i in range(1, kp.n + 1):
            for j in range(kp.W + 1):
                # Implement logic as described above
                if kp.w[i - 1] > j:
                    m[i][j] = m[i - 1][j]
                else:
                    m[i][j] = max(m[i - 1][j], m[i - 1][j - kp.w[i - 1]] + kp.v[i - 1])

        return m[kp.n][kp.W], self.get_dp_solution_indexes(kp, m)


class ZeroOneExhaustive(KnapsackSolver):
    """Solve 0-1 Knapsack Problem by exhaustive search."""

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by exhaustive search.

        Generates all subsets, finds the value,
        and checks if this value is greater than the current greatest value
        """

        max_value: int = 0
        best_item_list: list[int] = []
        # Generate subsets
        for binary_list in itertools.product(["0", "1"], repeat=kp.n):
            binary = "".join(binary_list)
            subset = [index for index in range(kp.n) if binary[index] == "1"]
            subset_value = sum(
                value for index, value in enumerate(kp.v) if index in subset
            )
            subset_weight = sum(
                weight for index, weight in enumerate(kp.w) if index in subset
            )
            if subset_value > max_value and subset_weight <= kp.W:
                max_value = subset_value
                best_item_list = subset

        return max_value, best_item_list


class ZeroOneRecursive(KnapsackSolver):
    """Solve 0-1 Knapsack Problem by recursion, without storing values in array."""

    def _recursive(self, i: int, j: int, kp: KnapsackProblem) -> int:
        """i: number of items to use, j: maximum weight."""
        if i == 0:
            return 0

        if kp.w[i - 1] > j:
            return self._recursive(i - 1, j, kp)
        else:
            return max(
                self._recursive(i - 1, j, kp),
                self._recursive(i - 1, j - kp.w[i - 1], kp) + kp.v[i - 1],
            )

    def _indexes_recursive(self, i: int, j: int, kp: KnapsackProblem) -> list[int]:
        if i == 0:
            return []

        if self._recursive(i, j, kp) > self._recursive(i - 1, j, kp):
            old_list = self._indexes_recursive(i - 1, j - kp.w[i - 1], kp)
            old_list.append(i - 1)
            return old_list
        else:
            return self._indexes_recursive(i - 1, j, kp)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by recursion (like DP), without storing values in array."""

        return self._recursive(kp.n, kp.W, kp), self._indexes_recursive(kp.n, kp.W, kp)


class ZeroOneRecursiveLRUCache(ZeroOneRecursive):
    """Solve 0-1 Knapsack Problem by recursion, without storing values in array."""

    @lru_cache # type: ignore # Doesn't work as only caches first call.
    def _recursive(self, i: int, j: int, kp: KnapsackProblem) -> int:
        """i: number of items to use, j: maximum weight."""
        if i == 0:
            return 0

        if kp.w[i - 1] > j:
            return self._recursive(i - 1, j, kp)
        else:
            return max(
                self._recursive(i - 1, j, kp),
                self._recursive(i - 1, j - kp.w[i - 1], kp) + kp.v[i - 1],
            )

    @lru_cache # type: ignore
    def _indexes_recursive(self, i: int, j: int, kp: KnapsackProblem) -> list[int]:
        if i == 0:
            return []

        if self._recursive(i, j, kp) > self._recursive(i - 1, j, kp):
            old_list = self._indexes_recursive(i - 1, j - kp.w[i - 1], kp)
            old_list.append(i - 1)
            return old_list
        else:
            return self._indexes_recursive(i - 1, j, kp)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by recursion (like DP), without storing values in array,
        but using lru_caching."""

        return self._recursive(kp.n, kp.W, kp), self._indexes_recursive(kp.n, kp.W, kp)
