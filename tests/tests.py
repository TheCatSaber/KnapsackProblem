import os
import sys
import unittest

from tester_base_classes import BaseZeroOneTester

# Put parent directory on import path, so can import stuff from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problems import KnapsackProblem
from solvers import (
    KnapsackSolver,
    ZeroOneDynamicProgrammingSolver,
    ZeroOneDynamicProgrammingSolverSlow,
)

# Tests ZOp01-ZOp08 from
# https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html
# Under LGPL (which is why this will be licensed under LGPL when you see it (just not yet
# in early commits because lazy))


class TestKnapsackProblemVerifierTest(unittest.TestCase):
    def test_n_below_zero(self):
        self.assertRaises(
            ValueError,
            KnapsackProblem,
            -1,
            [1],
            [1],
            5,
        )

    def test_w_wrong_length(self):
        self.assertRaises(
            ValueError,
            KnapsackProblem,
            5,
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            20,
        )

    def test_v_wrong_length(self):
        self.assertRaises(
            ValueError,
            KnapsackProblem,
            5,
            [1, 2, 3, 4, 5],
            [1, 2, 3],
            20,
        )

    def test_w_value_below_zero(self):
        self.assertRaises(
            ValueError,
            KnapsackProblem,
            2,
            [-1, 0],
            [0, 0],
            1,
        )

    def test_v_value_below_zero(self):
        self.assertRaises(
            ValueError,
            KnapsackProblem,
            2,
            [0, 0],
            [0, -1],
            1,
        )

    def test_w_below_zero(self):
        self.assertRaises(ValueError, KnapsackProblem, 1, [0], [0], -1)

    def test_normal_behavior_and_x_list(self):
        kp = KnapsackProblem(3, [1, 2, 3], [1, 2, 3], 5)
        self.assertEqual(kp.x, [0, 0, 0])


class TestKnapsackSolveBaseClass(unittest.TestCase):
    def test_weight_equals_zero(self):
        kp = KnapsackProblem(1, [0], [0], 5)
        self.assertRaises(
            ValueError, KnapsackSolver.check_strictly_positive, kp.w, kp.W, "Test"
        )

    def test_max_weight_equals_zero(self):
        kp = KnapsackProblem(1, [1], [1], 0)
        self.assertRaises(
            ValueError, KnapsackSolver.check_strictly_positive, kp.w, kp.W, "Test"
        )

    # Slightly contrived examples to check work
    def test_dp_solution_index_set_0_first(self):
        kp = KnapsackProblem(1, [2], [2], 0)
        m = [[0], [-1]]
        # Should update m
        KnapsackSolver.get_dp_solution_indexes(kp, m)
        self.assertEqual(m, [[0], [0]])

    def test_dp_solution_index_set_0_second(self):
        kp = KnapsackProblem(1, [1], [2], 1)
        m = [[-1, -1], [2, 2]]
        # Should update m
        KnapsackSolver.get_dp_solution_indexes(kp, m)
        self.assertEqual(m, [[-1, 0], [2, 2]])

    def test_dp_solution_indexes_error(self):
        kp = KnapsackProblem(3, [1, 2, 3], [3, 2, 1], 4)
        self.assertRaises(
            ValueError,
            KnapsackSolver.get_dp_solution_indexes,
            kp,
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
        )


class TestZeroOneDPSolver(BaseZeroOneTester):
    solver = ZeroOneDynamicProgrammingSolver()

    # Tests take ~ 30 seconds if p08 is included.
    def test_p08(self):
        self.skipTest("This algorithm is too slow to run this test.")


class TestZeroOneDPSolverSlow(BaseZeroOneTester):
    solver = ZeroOneDynamicProgrammingSolverSlow()

    # Tests take ~ 150 seconds if p08 is included.
    def test_p08(self):
        self.skipTest("This algorithm is too slow to run this test.")


unittest.main()
