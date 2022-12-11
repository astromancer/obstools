
# local
from scrawl.video import VideoDisplay

# relative
from ..campaign import ImageHDU


class FitsVideo(VideoDisplay):
    """Memory efficient video display for large FITS files."""
    def _check_data(self, data):
        assert isinstance(data, ImageHDU)
        assert data.ndim == 3
        
        return data.section, data.nframes
