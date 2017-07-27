import matplotlib.pyplot as plt
from obstools.plan.viz_blit import VisPlot
# from obstools.plan.viz import resolver
import time

if __name__ == '__main__':

    # t0 = time.time()
    # resolver('CTCV J1928-5001')
    # print('resolve took', time.time()-t0)
    #
    # t0 = time.time()
    # resolver('CTCV J1928-5001')
    # print('resolve took', time.time()-t0)

    #from .viz import VisPlot
    targets = ['CTCV J1928-5001',
            # 'IGR J14536-5522',
            'IGR J19552+0044',
            # 'QS Tel',
            # 'V1432 Aql',
            # 'HU Aqr',
            # 'CD Ind',
            'CRTS SSS100805 J194428-420209']

    viz = VisPlot(targets=targets)
    viz.plot_vis(sort=False)
    #viz.add_target('V895 Cen')

    viz.connect()

    plt.show()