import matplotlib.pyplot as plt
from obstools.plan.visibilities import VizPlot

if __name__ == '__main__':
    mcvs = ['AR Sco',
            'V895 Cen',
            'V895 Cen',
            'IGR J14536-5522',
            '1RXS J174320.1-042953',
            'RX J1610.1+0352',
            'RX J1745.5-2905',
            'SDSS J151415.65+074446.4',
            '2XMM J154305.5-522709',
            '2MASS J19283247-5001344',
            'QS Tel',
            'V1432 Aql',
            'CRTS SSS100805 J194428-420209',
            'IGR J19552+0044',
            'HU Aqr',
            'CD Ind',
            'CE Gru',
            'BL Hyi',
            'RX J0154.0-5947',
            'FL Cet',
            'FO Aqr']
    viz = VizPlot(mcvs)
    viz.plot_vis(cmap='jet')  # sort=False
    viz.add_target('V895 Cen')
    viz.connect()

    plt.show()
