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
from typing import Generator


from problems import KnapsackProblem

KnapsackSolution = tuple[int, list[int]]
MITM_subset = dict[str, tuple[int, int]]


class KnapsackSolver(ABC):
    """Abstract Base Class for Methods to solve Knapsack Problems
    defined by problems.KnapsackProblem objects.
    """

    @abstractmethod
    def __str__(self) -> str:
        """__str__ magic method"""

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
    def make_subsets(
        cls, kp: KnapsackProblem, start: int = 0, end: int = -1
    ) -> Generator[tuple[str, list[int], int, int], None, None]:
        """Make subsets of kp, or subsets of parition of kp.

        start (int): index of kp.w/kp.v to start at (default 0)
        end (int): index of kp.w/kp.v to end at +1
        (so range(start, end) would enumerate indexes of kp.w/kp.v to use), -1 => kp.n (default -1)
        index_offset
        """
        if end == -1:
            end = kp.n
        for binary_list in itertools.product(["0", "1"], repeat=(end - start)):
            binary = "".join(binary_list)
            subset = [
                index + start for index in range(end - start) if binary[index] == "1"
            ]
            subset_value = sum(
                value for index, value in enumerate(kp.v) if index in subset
            )
            subset_weight = sum(
                weight for index, weight in enumerate(kp.w) if index in subset
            )
            yield binary, subset, subset_value, subset_weight

    @classmethod
    def binary_to_solution(cls, binary: str) -> list[int]:
        if not all(char in "01" for char in binary):
            raise ValueError(
                "binary must be string of only 0 or qs in binary_to_solution"
            )
        return [int(char) for char in binary]


class BaseZeroOneDynamicProgramming(KnapsackSolver):
    """Defines solution methods used by both 0-1 Dynamic Programming algorithms
    - does not define a solution itself.
    """

    @abstractmethod
    def __str__(self) -> str:
        """__str__ magic method"""

    @abstractmethod
    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Abstract classmethod to sovle kp,
        returning maximum value and list of indices of items in optimal knapsack"""

    @classmethod
    def _recursive_index_dp(
        cls, i: int, j: int, m: list[list[int]], kp: KnapsackProblem
    ) -> list[int]:
        """Recursively get solution:
        if an item is introduced and value changes, that item is in the solution."""

        # Make solution set of 0s
        solution = [0 for _ in range(kp.n)]
        # Base case
        if i == 0:
            return solution

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
            solution = cls._recursive_index_dp(i - 1, j - kp.w[i - 1], m, kp)
            solution[i - 1] = 1  # i - 1 so index of item
            return solution
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


class ZeroOneDynamicProgrammingFast(BaseZeroOneDynamicProgramming):
    """Solve 0-1 Knapsack Problem, using Dynamic Programming,
    only calculating values needed (recursively), using second algorithm of
    https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Dynamic Programming (Fast)"

    def _recursive(
        self, i: int, j: int, m: list[list[int]], kp: KnapsackProblem
    ) -> int:
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


class ZeroOneDynamicProgrammingSlow(BaseZeroOneDynamicProgramming):
    """Solve 0-1 Knapsack Problem, using dynamic programming,
    calculating all values needed in array, using first algorithm of
    https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Dynamic Programming Slow"
    
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
    """Solve 0-1 Knapsack Problem by an exhaustive search
    of all 2^n subsets of the n items.
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Exhaustive"

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by exhaustive search.

        Generates all subsets, finds the value,
        and checks if this value is greater than the current greatest value
        """

        max_value: int = 0
        best_binary: str = ""
        # Generate subsets
        for binary, _, value, weight in self.make_subsets(kp):
            if value > max_value and weight <= kp.W:
                max_value = value
                best_binary = binary

        return max_value, self.binary_to_solution(best_binary)


class ZeroOneRecursive(KnapsackSolver):
    """Solve 0-1 Knapsack Problem by recursion, without storing values in an array."""

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Recursive"

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

        solution = [0 for _ in range(kp.n)]

        if i == 0:
            return solution

        if self._recursive(i, j, kp) > self._recursive(i - 1, j, kp):
            solution = self._indexes_recursive(i - 1, j - kp.w[i - 1], kp)
            solution[i - 1] = 1
            return solution
        else:
            return self._indexes_recursive(i - 1, j, kp)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by recursion (like DP), without storing values in array."""

        return self._recursive(kp.n, kp.W, kp), self._indexes_recursive(kp.n, kp.W, kp)


class ZeroOneRecursiveLRUCache(ZeroOneRecursive):
    """Solve 0-1 Knapsack Problem by recursion, without storing values in an array,
    but using the lru_cache decorator to cache the results of recursive calls.
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Recursive (LRU Cache)"

    @lru_cache  # type: ignore
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

    @lru_cache  # type: ignore
    def _indexes_recursive(self, i: int, j: int, kp: KnapsackProblem) -> list[int]:
        solution = [0 for _ in range(kp.n)]

        if i == 0:
            return solution

        if self._recursive(i, j, kp) > self._recursive(i - 1, j, kp):
            solution = self._indexes_recursive(i - 1, j - kp.w[i - 1], kp)
            solution[i - 1] = 1
            return solution
        else:
            return self._indexes_recursive(i - 1, j, kp)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem by recursion (like DP), without storing values in array,
        but using lru_caching."""

        return self._recursive(kp.n, kp.W, kp), self._indexes_recursive(kp.n, kp.W, kp)


class ZeroOneMeetInTheMiddle(KnapsackSolver):
    """Solve 0-1 Knapsack Problem using "meet-in-themiddle" algorithm.
    See https://en.wikipedia.org/wiki/Knapsack_problem#Meet-in-the-middle
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Meet-in-the-middle"

    @classmethod
    def _make_parition_subsets(
        cls, kp: KnapsackProblem
    ) -> tuple[MITM_subset, MITM_subset]:
        midpoint = kp.n // 2
        subsets_of_a = {
            binary: (value, weight)
            for binary, _, value, weight in cls.make_subsets(kp, end=midpoint)
        }
        subsets_of_b = {
            binary: (value, weight)
            for binary, _, value, weight in cls.make_subsets(kp, start=midpoint)
        }
        return subsets_of_a, subsets_of_b

    @classmethod
    def _compute_best_subset(
        cls,
        subsets_of_a: MITM_subset,
        subsets_of_b: MITM_subset,
        W: int,
        ordered_weights: bool = False,
    ) -> KnapsackSolution:
        max_value = 0
        best_binary = ""

        for binary, (subset_value, subset_weight) in subsets_of_a.items():
            for binary_b, (subset_value_b, subset_weight_b) in subsets_of_b.items():
                if subset_weight + subset_weight_b <= W:
                    combined_value = subset_value + subset_value_b
                    if combined_value > max_value:
                        max_value = combined_value
                        best_binary = "".join((binary, binary_b))
                elif ordered_weights:
                    # Weights in b gone over what this A subset can handle, so next A subset
                    break
        
        return max_value, cls.binary_to_solution(best_binary)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem using "meet-in-themiddle" algorithm.
        See https://en.wikipedia.org/wiki/Knapsack_problem#Meet-in-the-middle"""

        subsets_of_a, subsets_of_b = self._make_parition_subsets(kp)
        return self._compute_best_subset(subsets_of_a, subsets_of_b, kp.W)


class ZeroOneMeetInTheMiddleOptimised(ZeroOneMeetInTheMiddle):
    """Solve 0-1 Knapsack Problem using "meet-in-themiddle" algorithm,
    but optimised to remove unnecessary subsets and reduce run-time.
    See https://en.wikipedia.org/wiki/Knapsack_problem#Meet-in-the-middle
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Meet-in-the-middle (optimised)"

    @classmethod
    def _optimise_subsets_of_b(cls, subsets_of_b: MITM_subset) -> MITM_subset:
        # Sort by weight
        subsets_of_b = {
            binary: (value, weight)
            for binary, (value, weight) in sorted(
                subsets_of_b.items(), key=lambda item: item[1][1]
            )
        }
        # Discard if this item weighs more than another subset with greater or equal value.
        # (so discarded weighs more and has a lower or equal value)
        new_subsets: MITM_subset = {}
        for binary, (value, weight) in subsets_of_b.items():  # Starts at lowest weight
            for (accepted_value, accepted_weight) in new_subsets.values():
                if weight > accepted_weight and value <= accepted_value:
                    # Discard
                    break
            else:  # No break (item not discarded)
                new_subsets[binary] = (value, weight)
        subsets_of_b = new_subsets
        return subsets_of_b

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve 0-1 Knapsack Problem using "meet-in-themiddle" algorithm,
        using optimisation steps therein described.
        See https://en.wikipedia.org/wiki/Knapsack_problem#Meet-in-the-middle"""

        subsets_of_a, subsets_of_b = self._make_parition_subsets(kp)
        subsets_of_b = self._optimise_subsets_of_b(subsets_of_b)
        return self._compute_best_subset(
            subsets_of_a, subsets_of_b, kp.W, ordered_weights=True
        )
