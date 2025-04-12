from flight_manger.cost_functions import (
    IDEAL_ALTITUDE,
    __altitude_reward,
    __pitch_reward,
    __roll_reward,
)


def test_pitch_reward():
    assert __pitch_reward(0) == 0
    assert __pitch_reward(90) == -1
    assert __pitch_reward(-90) == -1
    assert __pitch_reward(45) == -0.5


def test_roll_reward():
    assert __roll_reward(0) == 0
    assert __roll_reward(180) == -1
    assert __roll_reward(-180) == -1


def test_altitude_reward():
    assert __altitude_reward(IDEAL_ALTITUDE) == 0
    assert __altitude_reward(IDEAL_ALTITUDE / 2) == -0.5
    assert __altitude_reward(0) == -1
