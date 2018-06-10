#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import math
import pytest

class TestClass(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_name(self):
        pass

    @pytest.mark.skip(reason="skip it for a moment")
    def test_gradiant_math(self):

        # formula

        x, y = 3, 2
        u = 0.1
        z = pow(x, 2) + pow(y, 2)


        print(z)
