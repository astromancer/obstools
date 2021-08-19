
from scrawl.imagine import VideoDisplay
from ..campaign import HDUExtra


class FitsVideo(VideoDisplay):
    """Memory efficient video display for large FITS files."""
    def _check_data(self, data):
        assert isinstance(data, HDUExtra)
        assert data.ndim == 3
        
        return data.section, data.nframes