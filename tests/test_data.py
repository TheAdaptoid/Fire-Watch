import numpy as np
import pytest
import torch

from flight_manger.data import Observation

ASL: float = 7144.2
PITCH_ANGLE: float = 12.5
ROLL_ANGLE: float = 5.5
HEADING_DIFFERENCE: float = -0.78


class TestObservation:

    def test_initialization(self):
        observation = Observation(
            altitude_asl=ASL,
            pitch_angle=PITCH_ANGLE,
            roll_angle=ROLL_ANGLE,
            heading_offset=HEADING_DIFFERENCE,
        )

        assert observation.altitude_asl == ASL
        assert observation.pitch_angle == PITCH_ANGLE
        assert observation.roll_angle == ROLL_ANGLE
        assert observation.heading_offset == HEADING_DIFFERENCE

    def test_normalize_values(self):
        observation = Observation(
            altitude_asl=ASL,
            pitch_angle=PITCH_ANGLE,
            roll_angle=ROLL_ANGLE,
            heading_offset=HEADING_DIFFERENCE,
        )

        observation.normalize_values()

        assert observation.altitude_asl >= 0
        assert observation.altitude_asl <= 1
        assert observation.pitch_angle >= -1
        assert observation.pitch_angle <= 1
        assert observation.roll_angle >= -1
        assert observation.roll_angle <= 1
        assert observation.heading_offset >= -1
        assert observation.heading_offset <= 1

    def test_to_tensor(self):
        observation = Observation(
            altitude_asl=ASL,
            pitch_angle=PITCH_ANGLE,
            roll_angle=ROLL_ANGLE,
            heading_offset=HEADING_DIFFERENCE,
        )

        tensor = observation.to_tensor()

        assert tensor.shape == (4,)
        assert tensor.dtype == torch.float32

        assert tensor[0] == ASL
        assert tensor[1] == PITCH_ANGLE
        assert tensor[2] == ROLL_ANGLE
        assert tensor[3] == HEADING_DIFFERENCE

    def test_to_ndarray(self):
        observation = Observation(
            altitude_asl=ASL,
            pitch_angle=PITCH_ANGLE,
            roll_angle=ROLL_ANGLE,
            heading_offset=HEADING_DIFFERENCE,
        )

        ndarray = observation.to_ndarray()

        assert ndarray.shape == (4,)
        assert ndarray.dtype == np.float32

        assert ndarray[0] == ASL
        assert ndarray[1] == PITCH_ANGLE
        assert ndarray[2] == ROLL_ANGLE
        assert ndarray[3] == HEADING_DIFFERENCE

    def test_to_tuple(self):
        observation = Observation(
            altitude_asl=ASL,
            pitch_angle=PITCH_ANGLE,
            roll_angle=ROLL_ANGLE,
            heading_offset=HEADING_DIFFERENCE,
        )

        tuple_ = observation.to_tuple()

        assert len(tuple_) == 4
        assert type(tuple_[0]) == float

        assert tuple_[0] == ASL
        assert tuple_[1] == PITCH_ANGLE
        assert tuple_[2] == ROLL_ANGLE
        assert tuple_[3] == HEADING_DIFFERENCE

    def test_length(self):
        observation = Observation(
            altitude_asl=ASL,
            pitch_angle=PITCH_ANGLE,
            roll_angle=ROLL_ANGLE,
            heading_offset=HEADING_DIFFERENCE,
        )

        assert observation.length == 4
