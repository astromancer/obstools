from obstools.image.noise import CCDNoiseModel
from astropy import units


class TestCCDNoiseModel:
    def test_init(self):
        noise = CCDNoiseModel(gain=2.3, readout=12, sky=101.2)
        assert noise.sky == 44 * units.adu
        assert noise.total_var == (113.2 / 2.3)

    def test_init(self):
        noise = CCDNoiseModel(gain=2.3, readout=12, sky=101.2)
        assert noise.sky == 44 * units.adu
        assert noise.total_var == (113.2 / 2.3)
