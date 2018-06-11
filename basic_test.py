#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import logistic

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def test_description():
    # 은닉층 1
    w1 = np.array([[3.214, -4.562, -0.541, -2.359, -1.071, -2.808, -1.382, 3.991, -2.218, 5.730, 5.310, -2.286]], dtype=float)

    # 학습데이터
    """
    1 1 1
    1 0 1   # 숫자 0 처럼 보이는 비트맵
    1 0 1
    1 1 1
    """
    x = np.array([[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=float)

    b1 = -1.076

    assert len(w1) == 1
    assert len(x) == 1
    # 1 * 12    ** 12 * 1 = 1

    z21 = np.dot(w1, x.T) + b1
    a21 = sigmoid(z21)

    print("sigmoid! 출력! a21: {}".format(sigmoid(z21)))
    print("scipy sigmoid a21: {}".format(logistic.cdf(z21)))


    w2 = np.array([[-4.044, -3.275, 1.017, -1.716, 5.457, -1.821, 5.361, 0.303, 0.940, -0.289, 3.505, 1.463]])
    b2 = 0.687

    z22 = np.dot(w2, x.T) + b2
    a22 = sigmoid(z22)

    # 해당 입력값을 시그모이드로 돌린다!
    print("sigmoid! 출력! a22: {}".format(sigmoid(z22)))
    print("scipy sigmoid a22: {}".format(logistic.cdf(z22)))

    w3 = np.array([-1.712, 3.601, -0.774, -1.456, -0.836, -2.440, 1.496, -0.193, 3.128, 0.423, -3.249, 2.292])
    b3 = -1.189


    z23 = np.dot(w3, x.T) + b3
    a23 = sigmoid(z23)

    print("sigmoid! 출력! a22: {}".format(sigmoid(z23)))
    print("scipy sigmoid a22: {}".format(logistic.cdf(z23)))

    # 출력층 w 입니다.
    w41 = np.array([[-3.575, 4.446, 5.666]])
    b41 = -5.578

    # 출력층이기 때문에 앞서 출력값들이 다시 입력값이 되어버립니다!.
    z31 = np.dot(w41, np.array([a21, a22, a23])) + b41
    a31 = sigmoid(z31)

    print("sigmoid! 출력! a31: {}".format(sigmoid(z31)))
    print("scipy sigmoid a31: {}".format(logistic.cdf(z31)))


    w42 = np.array([[-0.9406, 2.93089, -3.4101]])
    b42 = -2.2691

    z32 = np.dot(w42, np.array([a21, a22, a23])) + b42
    a32 = sigmoid(z32)

    print("sigmoid! 출력! a32: {}".format(sigmoid(z32)))
    print("scipy sigmoid a32: {}".format(logistic.cdf(z32)))

    t11 = 1
    t12 = 0

    c1 = 0.5 * (pow(t11 - a31, 2) + pow(t12 - a32, 2))
    # c1 ~ 0.046xxxx값이 나와야한다.


def test_simple_basic_test():
    # 은닉층 1
    w1 = np.array([[3.214, -4.562, -0.541, -2.359, -1.071, -2.808, -1.382, 3.991, -2.218, 5.730, 5.310, -2.286]], dtype=float)

    # 학습데이터
    """
    0 1 1
    1 0 1   # 숫자 0 처럼 보이는 비트맵
    1 0 1
    1 1 1
    """
    # x = np.array([[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=float)
    x = np.array([[0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=float)

    b1 = -1.076

    assert len(w1) == 1
    assert len(x) == 1
    # 1 * 12    ** 12 * 1 = 1

    z21 = np.dot(w1, x.T) + b1
    a21 = sigmoid(z21)

    print("sigmoid! 출력! a21: {}".format(sigmoid(z21)))
    print("scipy sigmoid a21: {}".format(logistic.cdf(z21)))


    w2 = np.array([[-4.044, -3.275, 1.017, -1.716, 5.457, -1.821, 5.361, 0.303, 0.940, -0.289, 3.505, 1.463]])
    b2 = 0.687

    z22 = np.dot(w2, x.T) + b2
    a22 = sigmoid(z22)

    # 해당 입력값을 시그모이드로 돌린다!
    print("sigmoid! 출력! a22: {}".format(sigmoid(z22)))
    print("scipy sigmoid a22: {}".format(logistic.cdf(z22)))

    w3 = np.array([-1.712, 3.601, -0.774, -1.456, -0.836, -2.440, 1.496, -0.193, 3.128, 0.423, -3.249, 2.292])
    b3 = -1.189


    z23 = np.dot(w3, x.T) + b3
    a23 = sigmoid(z23)

    print("sigmoid! 출력! a22: {}".format(sigmoid(z23)))
    print("scipy sigmoid a22: {}".format(logistic.cdf(z23)))

    # 출력층 w 입니다.
    w41 = np.array([[-3.575, 4.446, 5.666]])
    b41 = -5.578

    # 출력층이기 때문에 앞서 출력값들이 다시 입력값이 되어버립니다!.
    z31 = np.dot(w41, np.array([a21, a22, a23])) + b41
    a31 = sigmoid(z31)

    print("sigmoid! 출력! a31: {}".format(sigmoid(z31)))
    print("scipy sigmoid a31: {}".format(logistic.cdf(z31)))


    w42 = np.array([[-0.9406, 2.93089, -3.4101]])
    b42 = -2.2691

    z32 = np.dot(w42, np.array([a21, a22, a23])) + b42
    a32 = sigmoid(z32)

    print("sigmoid! 출력! a32: {}".format(sigmoid(z32)))
    print("scipy sigmoid a32: {}".format(logistic.cdf(z32)))

    t11 = 1
    t12 = 0

    c1 = 0.5 * (pow(t11 - a31, 2) + pow(t12 - a32, 2))
    # c1 ~ 0.046xxxx값이 나와야한다.



    pass
