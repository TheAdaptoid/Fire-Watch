from flight_manger.utils import (
    absolute_negation,
    negative_to_positive,
    normalize_value,
    zero_to_one,
)


def test_normalize_value():
    assert normalize_value(0, 0, 10) == 0
    assert normalize_value(5, 0, 10) == 0.5
    assert normalize_value(10, 0, 10) == 1


def test_absolute_negation():
    assert absolute_negation(0) == 0
    assert absolute_negation(5) == -5
    assert absolute_negation(-5) == -5
    assert absolute_negation(-0.5) == -0.5


def test_zero_to_one():
    assert zero_to_one(0, 0, 10) == 0
    assert zero_to_one(5, 0, 10) == 0.5
    assert zero_to_one(10, 0, 10) == 1


def test_negative_to_positive():
    assert negative_to_positive(0, 0, 10) == -1
    assert negative_to_positive(5, 0, 10) == 0
    assert negative_to_positive(10, 0, 10) == 1
