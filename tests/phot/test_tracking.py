

def test(hdu):
    #memloc = Path(tracker.xy_offsets.filename).parent
    n = 10
    tracker, reg = SourceTracker.from_hdu(hdu)

    # xy lower left corner first frame wrt global seg
    llc = reg.xy_offsets.min(0).round().astype(int)
    self = tracker = SourceTracker(reg.xy, reg.global_seg(), llc=llc)
    tracker.init_mem(n)

    im = tracker.plot(hdu.calibrated[0])
    im.ax.plot(*tracker.coords.T, 'rx')

    #
    seg = self.seg.select_region(-llc[::-1], image.shape)
    lines = seg.draw_contours(im.ax, color='g')


    image = hdu.oriented[0]
    xy = self.measure_source_locations(image, None, self.llc)

    im.ax.plot(*xy.T, 'gx')