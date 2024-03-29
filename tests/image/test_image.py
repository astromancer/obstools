# std
import textwrap as txw

# third-party
import pytest
import numpy as np

# local
from obstools.image import Image, SkyImage


# ---------------------------------------------------------------------------- #


class TestImage:
    def test_init(self):
        data = np.random.randn(10, 10)
        img = Image(data)
        assert (img.data == data).all()

    def test_repr(self, image):
        assert repr(image) == '<Image(shape=(10, 10))>'


class TestSkyImage:
    def test_init(self):
        data = np.random.randn(10, 10)
        img = SkyImage(data, (1, 1))
        assert (img.data == data).all()

        with pytest.raises(ValueError):
            SkyImage(data)

    def test_detect(self, skyimage0):
        skyimage0.detect()
        assert len(skyimage0.xy)
        assert len(skyimage0.counts)

    @pytest.mark.mpl_image_compare(baseline_dir='data/images',
                                   remove_text=True)
    def test_plot(self, skyimage0):
        skyimage0.angle = np.pi / 12
        display, art = skyimage0.plot()
        return art.image.figure

    def test_repr(self, skyimage0):
        assert (repr(skyimage0) == txw.dedent('''\
                <SkyImage(shape=(128, 128),
                          scale=array([0.01007813, 0.01007813]),
                          origin=array([0., 0.]),
                          angle=0.0)>'''))
        
    def test_copy(self, skyimage0):
        clone = skyimage0.copy()
        assert skyimage0 == clone
        
    def test_calibration(self, skyimage0):
        # SkyImage(skyimage0)
        skyimage0.set_calibrators(dark=10 * np.ones(skyimage0.shape),
                                  flat=2 * np.ones(skyimage0.shape),
                                  gain=7)

        ref =  (skyimage0.data - 10) / 2 * 7
        assert np.allclose(skyimage0[:], ref)
        
        # test clone
        clone = skyimage0.copy()
        assert np.allclose(clone[:], ref)

            