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
        """Abstract method to solve kp (problems.KnapsackProblem).

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).
        """

    # Things used by multiple sub-algorithms.
    @classmethod
    def check_strictly_positive(cls, w: list[int], W: int, error_name: str) -> None:
        """Checks that all weights in w (list[int]) and
        that W itself (int) are strictly positive (>0).
        If so, return None, otherwise raise ValueError.
        """
        if any(i <= 0 for i in w):
            raise ValueError(f"{error_name} requires all weights to be greater than 0.")
        if W <= 0:
            raise ValueError(f"{error_name} requires max weights to be greater than 0.")
        return None

    @classmethod
    def make_subsets(
        cls, kp: KnapsackProblem, start: int = 0, end: int = -1
    ) -> Generator[tuple[str, list[int], int, int], None, None]:
        """Make subsets of kp (problems.KnapsackProblem), or subsets of partition of kp.

        Each subset is a tuple with a string (the binary representation of the subset),
        a list of integers (the indexes of the items in the subset), the value of the subset,
        and the weight of the subset.

        The subsets are returned within a generator object.

        Optional arguments:
        start (int): index of kp.w/kp.v to start at (default 0).
        end (int): index of kp.w/kp.v to end at +1: -1 means end at the last item (default -1).
        start and end are such that range(start, end) would list the indexes in the subset.
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
        """Convert a string version of a binary to a list of ints version.
        
        Binary (str): string of "0"s and "1"s.
        """
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
        """Abstract method to solve kp (problems.KnapsackProblem).

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).
        """

    @classmethod
    def _recursive_index_dp(
        cls, i: int, j: int, m: list[list[int]], kp: KnapsackProblem
    ) -> list[int]:
        """Recursively get solution: if an item is introduced
        and the value stored in m changes, that item is in the solution.

        Should be interally called from other methods using i = kp.n, and j = kp.W.

        Return solution list of 1 if item is in the solution, 0 otherwise.

        Arguments
        i (int): first index in m (number of items to use in partial solution).
        j (int): second index in m (maximum weight of partial solution).
        m (list[list[int]]): 2D list of ints, representing the maximum value
        obtainable with a partial solution using i items and a maximum weight of j
        (only cells with i == 0 or j == 0 may be not set with a value of -1,
        as these must be 0).
        kp (problems.KnapsackProblem): the Knapsack Problem that is being solved.
        """

        # See https://github.com/thecatsaber/knapsackproblem#Solution_items
        # for a full explanation.
        # Make solution set of 0s.
        solution = [0 for _ in range(kp.n)]
        # Part 1: Base case.
        if i == 0:
            return solution

        # Check for value being -1 (since certain algorithms set values to be -1 as default)
        # If the array value is on the edge of the array (i == 0 or j == 0),
        # it should be 0, so set it to this
        # (Part 1 of https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation).
        # Otherwise raise ValueError, as unsure what it should be.
        if m[i][j] == -1 and (i == 0 or j == 0):
            m[i][j] = 0
        if m[i - 1][j] == -1 and (i - 1 == 0 or j == 0):
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
        # Part 2: Item in solution if value changes when item introduced.
        if m[i][j] > m[i - 1][j]:
            solution = cls._recursive_index_dp(i - 1, j - kp.w[i - 1], m, kp)
            solution[i - 1] = 1  # i - 1 so index of item.
            return solution
        # Part 3: Item not in solution.
        else:
            return cls._recursive_index_dp(i - 1, j, m, kp)

    @classmethod
    def get_dp_solution_indexes(
        cls, kp: KnapsackProblem, m: list[list[int]]
    ) -> list[int]:
        """Get binary list showing which items are in the solution,
        using the BaseZeroOneDynamicProgramming._recursive_index_dp
        method (middle-manager between solve methods and this method).

        Return what is returned by this method (i.e. a list of ints: 1 if item
        is in the solution, 0 otherwise).

        kp (problems.KnapsackProblem): the Knapsack Problem that is being solved.
        m (list[int[int]]): 2D list of ints, representing the maximum value
        obtainable with a partial solution using i items and a maximum weight of j
        (only cells with i == 0 or j == 0 may be not set with a value of -1,
        as these must be 0).
        """

        return cls._recursive_index_dp(kp.n, kp.W, m, kp)


class ZeroOneDynamicProgrammingFast(BaseZeroOneDynamicProgramming):
    """Solve 0-1 Knapsack Problem, using Dynamic Programming,
    only calculating values needed (recursively), using second algorithm of
    https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem.
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Dynamic Programming (Fast)"

    def _recursive(
        self, i: int, j: int, m: list[list[int]], kp: KnapsackProblem
    ) -> int:
        """Recursively generate solution to kp (problems.KnapsackProblem),
        returning maximum value (int) that can be achieved using i (int) items and a maximum
        weight of j (int).
         
        Should be interally called from other methods using i = kp.n, and j = kp.W.

        m (list[list[int]]): 2D list of ints, representing the maximum value
        obtainable with a partial solution using i items and a maximum weight of j.
        """
        # See https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation
        # for a full explanation.
        # Part 1: No items or no weight: no value.
        if i == 0 or j == 0:
            m[i][j] = 0
            return 0
        # If m[i - 1][j] not calculated, calculate it.
        if m[i - 1][j] == -1:
            m[i - 1][j] = self._recursive(i - 1, j, m, kp)
        # Part 2: Can item possibly fit in knapsack?
        if kp.w[i - 1] > j:
            m[i][j] = m[i - 1][j]
        else:
            # If m[i - 1][j - kp.w[i - 1]] not calculated, calculate it.
            if m[i - 1][j - kp.w[i - 1]] == -1:
                m[i - 1][j - kp.w[i - 1]] = self._recursive(
                    i - 1, j - kp.w[i - 1], m, kp
                )
            # Part 3: Choose highest value of including item and not including item.
            m[i][j] = max(m[i - 1][j], m[i - 1][j - kp.w[i - 1]] + kp.v[i - 1])
        return m[i][j]

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve kp (problems.KnapsackProblem) by Dynamic Programming.
        Recursively calculate values needed in m, starting at m[n][W].

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution). 

        See https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation
        for an explanation of the recursion used.
        """
        # Base case.
        if kp.n == 0:
            return 0, []

        # This algorithm assumes w1, w2, ... wn, W > 0, so test this.
        self.check_strictly_positive(kp.w, kp.W, "ZeroOneDynamicProgrammingSolverFast")

        # Create array with -1 so self._recursive can know whether a value has been set yet.
        m = [[-1 for _ in range(kp.W + 1)] for _ in range(kp.n + 1)]

        return self._recursive(kp.n, kp.W, m, kp), self.get_dp_solution_indexes(kp, m)


class ZeroOneDynamicProgrammingSlow(BaseZeroOneDynamicProgramming):
    """Solve 0-1 Knapsack Problem, using dynamic programming,
    calculating all values needed in array, using first algorithm of
    https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem.
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Dynamic Programming Slow"
    
    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve kp (problems.KnapsackProblem) by Dynamic Programming.
        Calculate all values in array.

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).

        See https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation
        for an explanation of how the values are values are set.
.
        """
        if kp.n == 0:
            return 0, []

        # This algorithm assumes w1, w2, ... wn, W > 0, so test this.
        self.check_strictly_positive(kp.w, kp.W, "ZeroOneDynamicProgrammingSolverSlow")

        # Creating with 0s so do not have to set top row and left column manually.
        # This covers Part 1: No items or no weight: no value.
        m = [[0 for _ in range(kp.W + 1)] for _ in range(kp.n + 1)]

        for i in range(1, kp.n + 1):
            for j in range(kp.W + 1):
                # Part 2: Can item possibly fit in knapsack?
                if kp.w[i - 1] > j:
                    m[i][j] = m[i - 1][j]
                else:
                    # Part 3: Choose highest value of including item and not including item.
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
        """Solve kp (problems.KnapsackProblem) by exhaustive search.

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).

        Generates all subsets, finds the value,
        and checks if this value is greater than the current greatest value.
        """

        max_value: int = 0
        best_binary: str = ""
        # Generate subsets.
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
        """Recursively generate solution to kp (problems.KnapsackProblem),
        returning maximum value (int) that can be achieved using i (int) items and a maximum
        weight of j (int).

        Should be interally called from other methods using i = kp.n, and j = kp.W.
        """
        # See https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation
        # for a full explanation.
        # Part 1: No items or no weight: no value.
        if i == 0 or j == 0:
            return 0

        # Part 2: Can item possibly fit in knapsack?
        if kp.w[i - 1] > j:
            return self._recursive(i - 1, j, kp)
        else:
            # Part 3: Choose highest value of including item and not including item.
            return max(
                self._recursive(i - 1, j, kp),
                self._recursive(i - 1, j - kp.w[i - 1], kp) + kp.v[i - 1],
            )

    def _indexes_recursive(self, i: int, j: int, kp: KnapsackProblem) -> list[int]:
        """Recursively get whether get index is in the solution or not.

        If an item is introduced and the value returned by ZeroOneRecursive._recursive changes,
        then that item is in the solution.

        Return solution list of 1 if item is in the solution, 0 otherwise.

        Should be interally called from other methods using i = kp.n, and j = kp.W.

        i (int): number of items in the partial solution.
        j (int): maximum weight of the partial solution.
        kp (problems.KnapsackProblem): the Knapsack Problem that is being solved.
        """
        # See https://github.com/thecatsaber/knapsackproblem#Solution_items
        # for a full explanation.
        # Make solution set of 0s.
        solution = [0 for _ in range(kp.n)]

        # Part 1: Base case.
        if i == 0:
            return solution

        # Part 2: Item in solution if value changes when item introduced.
        if self._recursive(i, j, kp) > self._recursive(i - 1, j, kp):
            solution = self._indexes_recursive(i - 1, j - kp.w[i - 1], kp)
            solution[i - 1] = 1
            return solution
        # Part 3: Item not in solution.
        else:
            return self._indexes_recursive(i - 1, j, kp)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve kp (problems.KnapsackProblem) by recursion, without using an array.

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).

        See https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation
        for an explanation of the recursion used.
        """

        # This algorithm assumes w1, w2, ... wn, W > 0, so test this.
        self.check_strictly_positive(kp.w, kp.W, "ZeroOneRecursive")

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
        """Recursively generate solution to kp (problems.KnapsackProblem),
        returning maximum value (int) that can be achieved using i (int) items and a maximum
        weight of j (int).

        Exactly the same internals as ZerOneRecursive._recursive, except decoratored
        with functools.lru_cache

        Should be interally called from other methods using i = kp.n, and j = kp.W.
        """
        # See https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation
        # for a full explanation.
        # Part 1: No items or no weight: no value.
        if i == 0 or j == 0:
            return 0

        # Part 2: Can item possibly fit in knapsack?
        if kp.w[i - 1] > j:
            return self._recursive(i - 1, j, kp)
        else:
            # Part 3: Choose highest value of including item and not including item.
            return max(
                self._recursive(i - 1, j, kp),
                self._recursive(i - 1, j - kp.w[i - 1], kp) + kp.v[i - 1],
            )

    @lru_cache  # type: ignore
    def _indexes_recursive(self, i: int, j: int, kp: KnapsackProblem) -> list[int]:
        """Recursively get whether get index is in the solution or not.

        Exactly the same internals as ZerOneRecursive._recursive, except decoratored
        with functools.lru_cache

        If an item is introduced and the value returned by ZeroOneRecursive._recursive changes,
        then that item is in the solution.

        Return solution list of 1 if item is in the solution, 0 otherwise.

        Should be interally called from other methods using i = kp.n, and j = kp.W.

        i (int): number of items in the partial solution.
        j (int): maximum weight of the partial solution.
        kp (problems.KnapsackProblem): the Knapsack Problem that is being solved.
        """
        # See https://github.com/thecatsaber/knapsackproblem#Solution_items
        # for a full explanation.
        # Make solution set of 0s.
        solution = [0 for _ in range(kp.n)]

        # Part 1: Base case.
        if i == 0:
            return solution

        # Part 2: Item in solution if value changes when item introduced.
        if self._recursive(i, j, kp) > self._recursive(i - 1, j, kp):
            solution = self._indexes_recursive(i - 1, j - kp.w[i - 1], kp)
            solution[i - 1] = 1
            return solution
        # Part 3: Item not in solution.
        else:
            return self._indexes_recursive(i - 1, j, kp)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve kp (problems.KnapsackProblem) by recursion, without using an array,
        but using the @functools.lru_cache decorator.
        
        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).

        See https://github.com/thecatsaber/knapsackproblem#Recursive_Explanation
        for an explanation of the recursion used.
        """

        # This algorithm assumes w1, w2, ... wn, W > 0, so test this.
        self.check_strictly_positive(kp.w, kp.W, "ZeroOneRecursiveLRUCache")

        return self._recursive(kp.n, kp.W, kp), self._indexes_recursive(kp.n, kp.W, kp)


class ZeroOneMeetInTheMiddle(KnapsackSolver):
    """Solve 0-1 Knapsack Problem using "meet-in-themiddle" algorithm.
    See https://en.wikipedia.org/wiki/Knapsack_problem#Meet-in-the-middle
    """

    def __str__(self) -> str:
        """__str__ magic method"""
        return "0-1 Meet-in-the-middle"

    @classmethod
    def _make_partition_subsets(
        cls, kp: KnapsackProblem
    ) -> tuple[MITM_subset, MITM_subset]:
        """Partition kp (problems.KnapsackProblem) into two subsets, A and B, and generate all
        possible subsets of these.

        Return 2 dicts (each dict is a MITM_subset; one for A and B respectively):
        the keys are the binary representations of these subsets,
        and the values are a tuple of the value and the weight of the corresponding subset.
        """
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
        """Find the best subset of the Knapsack Problem.

        Return the maximum value, and the solution in the list of 0s and 1s form.
        
        subsets_of_a (MITM_subset): 1st part of tuple produced by
        ZeroOneMeetInTheMiddle._make_partition_subsets.
        subsets_of_b (MITM_subset): 2nd part of tuple produced by
        ZeroOneMeetInTheMiddle._make_partition_subsets.
        W (int): maximum weight for the Knapsack Problem.
        ordered_weights: bool: if True, then subsets_of_b has been modified so the weights
        are ordered from lowest to highest, so the algorithm can skip the rest of the
        subsets_of_b once W has been exceeded.
        """
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
                    # Weights in b gone over what this A subset can handle, so next A subset.
                    break
        
        return max_value, cls.binary_to_solution(best_binary)

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve kp (problems.KnapsackProblem) using "meet-in-themiddle" algorithm.

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).

        See https://en.wikipedia.org/wiki/Knapsack_problem#Meet-in-the-middle.
        """

        subsets_of_a, subsets_of_b = self._make_partition_subsets(kp)
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
        """Optimise subsets_of_b (MITM_subset) by sorting by weight and discarding subsets
        that weigh more than another subset with a greater or equal value.
        
        Return the optimised subsets_of_b dict.
        """
        # Sort by weight.
        subsets_of_b = {
            binary: (value, weight)
            for binary, (value, weight) in sorted(
                subsets_of_b.items(), key=lambda item: item[1][1]
            )
        }
        # Discard if this item weighs more than another subset with a greater or equal value
        # (so discarded weighs more and has a lower or equal value).
        new_subsets: MITM_subset = {}
        for binary, (value, weight) in subsets_of_b.items():  # Starts at lowest weight.
            for (accepted_value, accepted_weight) in new_subsets.values():
                if weight > accepted_weight and value <= accepted_value:
                    # Discard
                    break
            else:  # No break (item not discarded).
                new_subsets[binary] = (value, weight)
        subsets_of_b = new_subsets
        return subsets_of_b

    def solve(self, kp: KnapsackProblem) -> KnapsackSolution:
        """Solve kp (problems.KnapsackProblem) using "meet-in-themiddle" algorithm,
        using optimisation steps therein described.

        Return maximum value and list indicating the items in the optimal knapsack
        (0: item is not in the solution; 1: item is in the solution).

        See https://en.wikipedia.org/wiki/Knapsack_problem#Meet-in-the-middle.
        """

        subsets_of_a, subsets_of_b = self._make_partition_subsets(kp)
        subsets_of_b = self._optimise_subsets_of_b(subsets_of_b)
        return self._compute_best_subset(
            subsets_of_a, subsets_of_b, kp.W, ordered_weights=True
        )
