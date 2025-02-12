# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0]


def f1(x: np.ndarray) -> np.ndarray:
    return x[0]


def f2(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.negative(np.add(np.add(np.sqrt(np.power(-0.106, x[0])), np.multiply(np.log2(-0.253), np.divide(x[1], 0.225))), np.sqrt(np.abs(np.power(-0.194, x[0]))))), np.abs(np.power(np.add(np.abs(np.negative(x[2])), np.abs(np.log10(-0.915))), np.multiply(np.minimum(np.tan(0.975), np.cos(0.651)), np.power(np.minimum(x[2], x[1]), np.add(-0.9, x[0]))))))


def f3(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.subtract(np.negative(np.add(x[2], np.log(np.divide(x[1], np.maximum(-0.874, -0.085))))), np.multiply(np.abs(np.multiply(np.add(np.subtract(0.314, 0.546), np.subtract(-0.126, x[0])), np.power(np.cos(-0.6276539976823008), np.multiply(x[2], 0.645)))), np.divide(np.abs(np.minimum(np.power(x[1], x[1]), np.subtract(x[0], 0.114))), np.multiply(np.tan(np.log10(-0.124)), np.abs(np.minimum(-0.935, x[1])))))), np.multiply(np.abs(np.multiply(x[1], np.negative(np.log10(np.subtract(0.935, x[1]))))), np.log2(np.maximum(np.log2(np.exp(np.negative(0.673))), x[1]))))


def f4(x: np.ndarray) -> np.ndarray:
    return np.add(np.minimum(np.maximum(np.negative(np.log2(np.divide(np.subtract(x[1], -0.909), np.log2(0.039)))), np.log10(np.abs(np.divide(np.minimum(-0.8152608049505645, 0.5591951904468146), np.maximum(0.035, 0.167))))), np.abs(np.multiply(np.log(np.add(np.add(-0.937, -0.78), np.abs(x[1]))), np.sin(np.maximum(np.sqrt(0.122), np.minimum(x[0], 0.082)))))), np.multiply(np.add(np.cos(np.divide(np.cos(np.minimum(0.38, x[1])), np.minimum(np.minimum(0.086, -0.211), np.maximum(-0.406, x[1])))), np.maximum(np.add(np.negative(np.power(-0.725, 0.174)), np.cos(np.negative(x[1]))), np.minimum(np.sin(np.negative(0.858)), np.add(np.multiply(x[1], -0.674), 0.884)))), np.exp(np.cos(np.tan(np.minimum(np.sin(x[1]), np.tan(-0.079)))))))


def f5(x: np.ndarray) -> np.ndarray:
    return 0.001


def f6(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.divide(x[1], np.cos(np.negative(-0.896))), x[0])


def f7(x: np.ndarray) -> np.ndarray:
    return np.abs(np.add(np.divide(np.multiply(np.exp(np.add(np.minimum(x[1], x[0]), np.abs(x[0]))), np.log(np.sqrt(np.divide(x[1], 0.135)))), np.exp(np.negative(np.log(np.multiply(x[1], x[0]))))), np.log10(np.negative(np.subtract(np.maximum(np.power(-0.371, 0.549), np.log10(x[0])), np.cos(np.add(-0.344, -0.975)))))))


def f8(x: np.ndarray) -> np.ndarray:
    return np.negative(np.multiply(np.subtract(np.negative(np.multiply(np.maximum(np.power(0.265, -0.028), np.negative(x[5])), np.subtract(np.power(np.power(-0.478, -0.028), np.abs(x[5])), np.log(0.0)))), np.abs(np.add(np.abs(np.log10(-0.266)), np.multiply(np.log10(-0.272), np.negative(x[5]))))), np.subtract(np.abs(np.minimum(np.log(np.multiply(0.064, 0.874)), np.multiply(np.add(-0.965, x[5]), np.log2(-0.521)))), np.subtract(np.subtract(np.abs(np.log2(-0.875)), np.subtract(np.tan(-0.68), np.add(0.533, 0.8204544064093195))), np.maximum(np.multiply(np.maximum(x[1], 0.64), np.log2(-0.296)), np.minimum(np.tan(-0.818), np.subtract(x[5], 0.135)))))))
