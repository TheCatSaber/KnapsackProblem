"""tester_base_classes.py: Base classes for tests."""
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


import unittest
from typing import Optional

from zero_one_problems import (
    ZOp,
    ZOp01,
    ZOp02,
    ZOp03,
    ZOp04,
    ZOp05,
    ZOp06,
    ZOp07,
    ZOp08,
    ZOpBaseCase,
    ZOpSimple,
    ZOpWP,
)

from context import KnapsackSolver


class BaseZeroOneTester(unittest.TestCase):
    """This is the base case, and will skip all tests."""

    solver: Optional[KnapsackSolver] = None

    def _test_case(self, p_a: ZOp):  # Problem, answer
        if self.solver is None:
            self.skipTest("No solver")
        else:
            self.assertEqual(self.solver.solve(p_a.p), p_a.a)

    def test_base_case(self):
        self._test_case(ZOpBaseCase)

    def test_simple(self):
        self._test_case(ZOpSimple)

    def test_wikipedia(self):
        self._test_case(ZOpWP)

    def test_p01(self):
        self._test_case(ZOp01)

    def test_p02(self):
        self._test_case(ZOp02)

    def test_p03(self):
        self._test_case(ZOp03)

    def test_p04(self):
        self._test_case(ZOp04)

    def test_p05(self):
        self._test_case(ZOp05)

    def test_p06(self):
        self._test_case(ZOp06)

    def test_p07(self):
        self._test_case(ZOp07)

    def test_p08(self):
        self._test_case(ZOp08)
