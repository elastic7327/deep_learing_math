#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import logistic
import numpy as np
import pytest

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def test_backpropatation():

    # n = 학습률
    n = 0.02

    w1 = np.array([[0.490, 0.348, 0.073, 0.837, -0.071, -3.617, -0.536, -0.023, -1.717, -1.456, -0.556, 0.852]], dtype=float)

    # 학습데이터
    """
    1 1 1
    1 0 1   # 숫자 0 처럼 보이는 비트맵
    1 0 1
    1 1 1
    """
    x = np.array([[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=float)

    b1 = -0.185

    assert len(w1) == 1
    assert len(x) == 1
    # 1 * 12    ** 12 * 1 = 1

    z21 = np.dot(w1, x.T) + b1
    a21 = sigmoid(z21)

    print("sigmoid! 출력! a21: {}".format(sigmoid(z21)))
    print("scipy sigmoid a21: {}".format(logistic.cdf(z21)))


    w2 = np.array([[0.442, -0.537, 1.008, 1.072, -0.733, 0.823, -0.453, -0.014, -0.027, -0.427, 1.876, -2.305]])
    b2 = 0.526

    z22 = np.dot(w2, x.T) + b2
    a22 = sigmoid(z22)

    # 해당 입력값을 시그모이드로 돌린다!
    print("sigmoid! 출력! a22: {}".format(sigmoid(z22)))
    print("scipy sigmoid a22: {}".format(logistic.cdf(z22)))

    w3 = np.array([0.654, -1.389, 1.246, 0.057, -0.183, -0.743, -0.461, 0.331, 0.449, -1.296, 1.569, -0.471])
    b3 = -1.169


    z23 = np.dot(w3, x.T) + b3
    a23 = sigmoid(z23)

    print("sigmoid! 출력! a22: {}".format(sigmoid(z23)))
    print("scipy sigmoid a22: {}".format(logistic.cdf(z23)))

    # 출력층 w 입니다.
    w41 = np.array([[0.388, 0.803, 0.029]])
    b41 = -1.438

    # 출력층이기 때문에 앞서 출력값들이 다시 입력값이 되어버립니다!.
    z31 = np.dot(w41, np.array([a21, a22, a23])) + b41
    a31 = sigmoid(z31)

    print("sigmoid! 출력! a31: {}".format(sigmoid(z31)))
    print("scipy sigmoid a31: {}".format(logistic.cdf(z31)))

    w42 = np.array([[-0.025, -0.790, 1.553]])
    b42 = -1.379

    z32 = np.dot(w42, np.array([a21, a22, a23])) + b42
    a32 = sigmoid(z32)

    print("sigmoid! 출력! a32: {}".format(sigmoid(z32)))
    print("scipy sigmoid a32: {}".format(logistic.cdf(z32)))

    t11 = 1
    t12 = 0

    c1 = 0.5 * (pow(t11 - a31, 2) + pow(t12 - a32, 2))
    print("c1: {}".format(c1))
    # c1 ~ 0.046xxxx값이 나와야한다.




