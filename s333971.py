# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(np.multiply(x[1], np.negative(-0.175)), np.abs(np.power(0.966, np.log2(np.abs(np.multiply(x[1], x[1])))))), x[0])


def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return np.add(np.negative(np.abs(np.log2(np.cos(np.minimum(np.log10(x[0]), np.multiply(-0.242, -0.028)))))), np.multiply(np.exp(np.divide(np.maximum(np.subtract(np.negative(0.028), np.log(0.121)), np.negative(np.abs(x[0]))), np.add(np.tan(np.sqrt(-0.228)), np.multiply(np.sqrt(-0.3470613249334904), np.add(x[0], -0.688))))), np.minimum(np.add(np.maximum(np.log2(np.add(x[0], x[1])), np.log2(np.minimum(x[2], x[0]))), np.negative(np.log2(np.subtract(0.612, x[0])))), np.add(np.divide(np.log10(np.multiply(-0.145, x[0])), np.maximum(np.maximum(-0.825, -0.112), np.minimum(x[2], -0.152))), np.add(np.log2(np.add(x[2], x[0])), np.add(np.divide(x[0], 0.42), np.cos(x[2])))))))


def f3(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.add(np.exp(np.divide(np.cos(np.add(np.exp(np.sqrt(np.maximum(np.sqrt(np.add(np.exp(np.sqrt(x[1])), x[1])), 0.556))), x[1])), 0.556)), np.log(np.minimum(np.power(np.exp(np.log2(np.cos(np.log(x[1])))), x[1]), x[1]))), x[1])


def f4(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.multiply(np.subtract(np.sqrt(np.abs(np.maximum(np.divide(x[1], x[1]), np.abs(x[1])))), np.subtract(np.power(np.abs(np.minimum(-0.385, -0.018)), np.abs(np.cos(-0.479804911416188))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.subtract(np.sin(np.negative(np.add(np.maximum(-0.571, -0.034), np.log(0.897)))), np.subtract(np.minimum(np.abs(np.subtract(x[1], -0.018)), np.sqrt(np.exp(0.9097510293420201))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.sqrt(np.negative(np.log10(-0.747))), np.subtract(np.abs(np.add(0.985, 0.942)), np.sqrt(np.multiply(x[1], x[1])))))))), np.log10(np.multiply(np.subtract(np.negative(np.abs(np.maximum(np.divide(x[1], 0.144), np.abs(x[1])))), np.subtract(np.power(np.abs(np.minimum(x[0], x[1])), np.tan(np.cos(-0.479804911416188))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.subtract(np.sin(np.negative(np.add(np.maximum(x[1], -0.034), np.log(0.897)))), np.subtract(np.minimum(np.abs(np.subtract(x[1], -0.018)), np.sqrt(np.exp(x[1]))), np.divide(0.977, np.negative(0.599)))), np.negative(np.multiply(np.sqrt(np.negative(np.log10(-0.747))), np.subtract(np.abs(np.add(0.985, 0.942)), np.sqrt(np.power(x[1], x[1]))))))))))


def f5(x: np.ndarray) -> np.ndarray:
    return np.divide(np.log2(np.exp(np.multiply(np.maximum(np.maximum(np.exp(0.95), np.add(x[1], -0.11213350080807727)), np.exp(np.cos(0.257))), np.multiply(np.power(np.log10(x[1]), np.divide(-0.874, -0.085)), x[1])))), np.divide(np.abs(np.divide(np.power(np.minimum(np.subtract(-0.968120953554666, x[1]), -0.005), np.subtract(np.negative(0.69), np.add(0.769, x[0]))), np.power(x[1], np.subtract(x[0], np.sqrt(x[0]))))), -0.6959832511287511))


def f6(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.add(np.multiply(np.divide(x[0], np.tan(np.sqrt(0.597))), np.negative(np.tan(0.597))), np.divide(x[1], np.cos(-0.909))), np.multiply(0.089, np.log2(np.exp(np.multiply(np.multiply(x[1], np.negative(np.abs(0.597))), np.maximum(-0.187, np.negative(-0.909)))))))


def f7(x: np.ndarray) -> np.ndarray:
    return np.exp(np.negative(np.negative(np.subtract(np.subtract(np.subtract(np.multiply(x[0], x[1]), np.multiply(-0.877, 0.959)), np.abs(np.maximum(0.713, x[1]))), np.minimum(np.log10(np.tan(0.034)), np.maximum(np.minimum(np.negative(np.negative(np.subtract(np.subtract(np.add(np.multiply(x[0], x[1]), np.divide(-0.057, 0.959)), np.exp(np.minimum(x[0], x[1]))), np.minimum(np.abs(np.log10(0.205)), np.power(np.power(-0.905, -0.527), np.cos(x[0])))))), -0.527), np.log10(np.divide(np.negative(np.negative(np.subtract(np.minimum(np.subtract(np.multiply(-0.836, 0.879), np.divide(-0.955, x[1])), np.exp(np.divide(x[0], 0.336))), np.maximum(np.log10(np.negative(-0.177)), np.power(np.power(-0.905, x[0]), np.log10(x[0])))))), x[1]))))))))


def f8(x: np.ndarray) -> np.ndarray:
    return np.divide(np.subtract(np.subtract(x[4], x[3]), np.add(np.negative(np.add(np.multiply(np.minimum(-0.921, -0.178), np.add(np.negative(np.subtract(np.power(np.divide(-0.275, -0.841), x[5]), np.log(np.multiply(-0.168, x[5])))), np.exp(x[5]))), np.sqrt(np.divide(np.subtract(np.multiply(np.negative(np.add(np.multiply(np.add(-0.921, x[4]), np.add(np.negative(np.subtract(np.divide(np.divide(-0.842, -0.841), 0.201), np.log(np.subtract(x[4], x[5])))), np.exp(np.subtract(np.power(x[4], np.multiply(np.power(0.448, 0.759), 0.148)), x[4])))), np.log2(np.divide(np.subtract(np.maximum(0.102, np.add(np.power(0.03, -0.601), x[0])), x[4]), 0.057)))), np.divide(np.add(0.448, -0.23), 0.148)), x[5]), 0.904)))), np.exp(x[5]))), -0.016)
