
# std
from pathlib import Path

# third-party
import pytest
import numpy as np

# local
from obstools.image import Image, SkyImage
# from pytest_steps import test_steps as steps


# ---------------------------------------------------------------------------- #
np.random.seed(999)

# ---------------------------------------------------------------------------- #

def load_test_data(filename):
    """load test data npz"""
    here = Path(__file__).parent
    # here = Path('/home/hannes/work/obstools/tests/')
    filename = (here / 'data' / filename).resolve()
    return list(np.load(filename).values())


images = load_test_data('images.npz')
fovs = load_test_data('fovs.npz')


# ---------------------------------------------------------------------------- #
# @pytest.fixture
# def random_skyimage():
#     return SkyImage(np.random.randn(10, 10), (1, 1))


# @pytest.fixture()
# def images():
#     return load_test_data('images.npz')


# @pytest.fixture
# def fovs():
#     return load_test_data('fovs.npz')


@pytest.fixture
def image():
    return Image(np.random.randn(10, 10))


@pytest.fixture
def skyimage(data, fov):
    return SkyImage(data, fov)


@pytest.fixture
def skyimage0():
    return SkyImage(images[0], fovs[0])
