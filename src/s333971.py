# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return np.add(x[0], np.log2(np.maximum(np.exp(np.log10(np.power(np.exp(np.log10(np.add(np.maximum(np.abs(0.9649573889319687), 0.6204281335809985), x[1]))), 0.969))), 0.969)))


def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return np.divide(np.add(np.subtract(x[0], -0.184), np.divide(np.divide(np.add(x[0], x[2]), 0.046), 0.046)), 0.046)


def f3(x: np.ndarray) -> np.ndarray:
    return np.negative(np.log2(np.log2(x[1])))


def f4(x: np.ndarray) -> np.ndarray:
    return np.exp(np.sin(0.7303678136165692))


def f5(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.subtract(np.negative(np.exp(np.abs(0.21031907159415808))), x[1]))


def f6(x: np.ndarray) -> np.ndarray:
    return np.add(x[1], x[1])


def f7(x: np.ndarray) -> np.ndarray:
    return np.negative(np.negative(np.subtract(np.exp(np.abs(x[0])), np.log2(np.abs(-0.043)))))


def f8(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.divide(np.log(np.minimum(x[4], x[1])), 0.038), x[2])
