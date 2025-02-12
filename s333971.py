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
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return np.divide(np.log2(np.divide(np.add(np.add(x[1], np.log2(np.exp(x[0]))), np.add(x[2], np.log10(np.exp(x[0])))), np.sqrt(np.subtract(np.log10(np.divide(-0.488, x[1])), np.add(np.power(x[0], -0.051), np.sqrt(x[0])))))), np.sin(np.sin(np.exp(np.log(np.log10(np.divide(-0.488, 0.804)))))))


def f3(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.subtract(np.negative(np.add(x[2], np.log(np.divide(x[1], np.maximum(-0.874, -0.085))))), np.multiply(np.abs(np.multiply(np.add(np.subtract(0.314, 0.546), np.subtract(-0.126, x[0])), np.power(np.cos(-0.6276539976823008), np.multiply(x[2], 0.645)))), np.divide(np.abs(np.minimum(np.power(x[1], x[1]), np.subtract(x[0], 0.114))), np.multiply(np.tan(np.log10(-0.124)), np.abs(np.minimum(-0.935, x[1])))))), np.multiply(np.abs(np.multiply(x[1], np.negative(np.log10(np.subtract(0.935, x[1]))))), np.log2(np.maximum(np.log2(np.exp(np.negative(0.673))), x[1]))))


def f4(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.multiply(np.subtract(np.sqrt(np.abs(np.maximum(np.divide(x[1], x[1]), np.abs(x[1])))), np.subtract(np.power(np.abs(np.minimum(-0.385, -0.018)), np.abs(np.cos(-0.479804911416188))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.subtract(np.sin(np.negative(np.add(np.maximum(-0.571, -0.034), np.log(0.897)))), np.subtract(np.minimum(np.abs(np.subtract(x[1], -0.018)), np.sqrt(np.exp(0.9097510293420201))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.sqrt(np.negative(np.log10(-0.747))), np.subtract(np.abs(np.add(0.985, 0.942)), np.sqrt(np.multiply(x[1], x[1])))))))), np.log10(np.multiply(np.subtract(np.negative(np.abs(np.maximum(np.divide(x[1], 0.144), np.abs(x[1])))), np.subtract(np.power(np.abs(np.minimum(x[0], x[1])), np.tan(np.cos(-0.479804911416188))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.subtract(np.sin(np.negative(np.add(np.maximum(x[1], -0.034), np.log(0.897)))), np.subtract(np.minimum(np.abs(np.subtract(x[1], -0.018)), np.sqrt(np.exp(x[1]))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.sqrt(np.negative(np.log10(-0.747))), np.subtract(np.abs(np.add(0.985, 0.942)), np.sqrt(np.power(x[1], x[1]))))))))))


def f5(x: np.ndarray) -> np.ndarray:
    return np.divide(np.log2(np.exp(np.multiply(np.maximum(np.maximum(np.exp(0.95), np.add(x[1], -0.11213350080807727)), np.exp(np.cos(0.257))), np.multiply(np.power(np.log10(x[1]), np.divide(-0.874, -0.085)), x[1])))), np.divide(np.abs(np.divide(np.power(np.minimum(np.subtract(-0.968120953554666, x[1]), -0.005), np.subtract(np.negative(0.69), np.add(0.769, x[0]))), np.power(x[1], np.subtract(x[0], np.sqrt(x[0]))))), -0.6959832511287511))


def f6(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.divide(x[1], np.cos(np.negative(-0.896))), x[0])


def f7(x: np.ndarray) -> np.ndarray:
    return np.abs(np.add(np.divide(np.multiply(np.exp(np.add(np.minimum(x[1], x[0]), np.abs(x[0]))), np.log(np.sqrt(np.divide(x[1], 0.135)))), np.exp(np.negative(np.log(np.multiply(x[1], x[0]))))), np.log10(np.negative(np.subtract(np.maximum(np.power(-0.371, 0.549), np.log10(x[0])), np.cos(np.add(-0.344, -0.975)))))))


def f8(x: np.ndarray) -> np.ndarray:
    return np.negative(np.multiply(np.subtract(np.negative(np.multiply(np.maximum(np.power(0.265, -0.028), np.negative(x[5])), np.subtract(np.power(np.power(-0.478, -0.028), np.abs(x[5])), np.log(0.0)))), np.abs(np.add(np.abs(np.log10(-0.266)), np.multiply(np.log10(-0.272), np.negative(x[5]))))), np.subtract(np.abs(np.minimum(np.log(np.multiply(0.064, 0.874)), np.multiply(np.add(-0.965, x[5]), np.log2(-0.521)))), np.subtract(np.subtract(np.abs(np.log2(-0.875)), np.subtract(np.tan(-0.68), np.add(0.533, 0.8204544064093195))), np.maximum(np.multiply(np.maximum(x[1], 0.64), np.log2(-0.296)), np.minimum(np.tan(-0.818), np.subtract(x[5], 0.135)))))))
