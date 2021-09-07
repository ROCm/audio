import torch
import unittest

from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda, skipIfRocm
from .functional_impl import Functional


@skipIfNoCuda
class TestFunctionalFloat32(Functional, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')

    @unittest.expectedFailure
    @skipIfRocm
    def test_lfilter_9th_order_filter_stability(self):
        super().test_lfilter_9th_order_filter_stability()


@skipIfNoCuda
@skipIfRocm
class TestLFilterFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')
