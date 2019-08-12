

# third-party libs
from grafico.multitab import MplMultiTab
from grafico.imagine import Compare3DImage

# local libs
from recipes.dict import TransDict



#****************************************************************************************************
class NullPSFPlot():
    def update(self, *args):
        pass

#****************************************************************************************************
#class LinkedAxesMixin():
    
    

#****************************************************************************************************        
class PSFPlot(Compare3DImage):
    '''Class for plotting / updating PSF models.'''
    #TODO: buttons for switching back and forth??
    pass
        
        
#****************************************************************************************************        
class MultiPSFPlot(MplMultiTab):
    '''Append each new fit plot as a figure in tab.'''
    #WARNING:  This is very slow!
    #TODO:  ui.show on first successful fit.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, X, Y, Z, data):
        plotter = PSFPlot()
        plotter.update(X, Y, Z, data)
        
        self.add_tab( plotter.fig )
    
#****************************************************************************************************        
class PSFPlotFactory():
    MODES = TransDict({ None       :      NullPSFPlot,
                        'update'   :      PSFPlot,
                        'append'   :      MultiPSFPlot } )
    MODES.add_translations( {False : None,
                             True : 'update'} )

    def __call__(self, mode):
        #if not mode in MODES.allkeys():
            #raise ValueError
            
        c = self.MODES.get(mode, NullPSFPlot)
        return c

psfPlotFactory = PSFPlotFactory()
