import matplotlib.pyplot as plt
from obstools.plan.viz import VisPlot


if __name__ == '__main__':
    #from .viz import VisPlot
    targets = ['CTCV J1928-5001',
            'IGR J14536-5522', 
            'IGR J19552+0044',
            'QS Tel', 
            'V1432 Aql',
            'HU Aqr', 
            'CD Ind',
            'CRTS SSS100805 J194428-420209']

    viz = VisPlot(targets=targets)
    viz.plot_vis(sort=False)
    #viz.add_target('V895 Cen')
    
    viz.connect()
    
    plt.show()