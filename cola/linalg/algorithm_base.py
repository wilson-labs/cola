from dataclasses import dataclass
from plum import parametric
from cola import LinearOperator
from cola.utils import export


@export
@dataclass
class Algorithm:
    pass


@parametric
class IterativeOperatorWInfo(LinearOperator):
    def __init__(self, A: LinearOperator, alg: Algorithm):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.alg = alg
        self.info = {}

    def _matmat(self, X):
        Y, self.info = self.alg(self.A, X)
        return Y

    def __str__(self):
        return f"{self.alg}({str(self.A)})"
