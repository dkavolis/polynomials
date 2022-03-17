#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Type

import pytest

try:
    from polynomials import Real
except ImportError:
    # using one of the builtin types, nothing to test
    pass
else:

    @pytest.mark.parametrize("number_type", (Real,))  # type: ignore
    class TestNumbers:
        def test_constructors_and_equality(self, number_type: Type[Any]):
            a = number_type(5)
            assert a == number_type(5)
            assert a == 5

            a = number_type(5.0)
            assert a == number_type(5.0)
            assert a == 5.0

            a = number_type("5.0")
            assert a == number_type(5)
            assert a == 5.0

        def test_operators(self, number_type: Type[Any]):
            a = number_type(5)

            # unary
            assert +a == 5
            assert -a == -5

            # binary same types
            assert a + a == 10
            assert a - a == 0
            assert a * a == 25
            assert a / a == 1

            # binary int
            assert a + 1 == 6
            assert 1 + a == 6
            assert a - 1 == 4
            assert 1 - a == -4
            assert a * 2 == 10
            assert 2 * a == 10
            assert a / 5 == 1
            assert 5 / a == 1

            # binary float
            assert a + 1.0 == 6.0
            assert 1.0 + a == 6.0
            assert a - 1.0 == 4.0
            assert 1.0 - a == -4.0
            assert a * 2.0 == 10.0
            assert 2.0 * a == 10.0
            assert a / 5.0 == 1.0
            assert 5.0 / a == 1.0

            # compound
            a += 1
            assert a == 6
            a -= 1
            assert a == 5
            a *= 2
            assert a == 10
            a /= 2
            assert a == 5

        @pytest.mark.parametrize("other_type", (None, int, float))
        def test_comparison(self, number_type: Type[Any], other_type: Type[Any]):
            if other_type is None:
                other_type = number_type

            a = number_type(5)
            b = other_type(10)
            assert a != b
            assert a < b
            assert b > a
            assert b >= a
            assert a <= b

        def test_functions(self, number_type: Type[Any]):
            a = number_type(-5)

            def test_fn(fn, expected, expect_type=number_type):
                b = fn(a)
                assert isinstance(b, expect_type)
                assert b == expected

            test_fn(abs, 5)
            test_fn(lambda x: pow(x, 2), 25)
            test_fn(int, -5, int)
            test_fn(float, -5, float)
            test_fn(hash, hash(a), int)

        def test_pickling(self, number_type: Type[Any]):
            import pickle

            a = number_type(5)
            assert pickle.loads(pickle.dumps(a)) == a
