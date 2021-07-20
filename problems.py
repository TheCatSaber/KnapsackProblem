class KnapsackProblem:
    """Class to store a knapsack problem."""

    @classmethod
    def verify(cls, n: int, w: list[int], v: list[int], W: int) -> None:
        """Verify Knapsack Problem is valid."""
        if n < 0:
            raise ValueError(
                "The number of items for the kanpsack problem must be greater than or"
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
                "The maximum weight (W) must be greater than or eual to 0."
            )

    def __init__(self, n: int, w: list[int], v: list[int], W: int) -> None:
        """Class to store a knapsack problem."""
        # verify raises errors and no return value, so just call it
        self.verify(n, w, v, W)
        self.n = n
        self.w = w
        self.v = v
        self.W = W
        # x is currently not used, but may be used in the future by unbounded/bounded problems
        # and if it is, then current programs should be adapted to use it.
        self.x = [0] * n  # Init list counting copies of each item to be used
