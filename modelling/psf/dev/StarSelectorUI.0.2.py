from mplMultiTab import *
from PyQt4.QtGui import QPushButton, QHBoxLayout, QSpacerItem, QPalette, QColor
from PyQt4 import QtCore

from matplotlib.colors import colorConverter
from string import Template

import numpy as np

from misc import make_ipshell
ipshell = make_ipshell()

class StarSelector(MplMultiTab):

    #====================================================================================================
    def create_main_frame(self, figures, labels):
        
        MplMultiTab.create_main_frame( self, figures, labels )
        
        buttonBox = self.create_buttons()
        
        ipshell()
        
        palette = QPalette(button.palette()) # make a copy of the palette
        palette.setColor(QPalette.ButtonText, QColor('black'))
        button.setPalette(palette)
        
        
        #self.connect(self.pushButton, SIGNAL('clicked()'), self.on_draw)
        
        
        # Other GUI controls
        #self.textbox = QLineEdit()
        #self.textbox.setMinimumWidth(200)
        #self.connect(self.textbox, SIGNAL('editingFinished ()'), self.on_draw)
        
        #self.draw_button = QPushButton("&Draw")
        #self.connect(self.draw_button, SIGNAL('clicked()'), self.on_draw)
        
        #self.grid_cb = QCheckBox("Show &Grid")
        #self.grid_cb.setChecked(False)
        #self.connect(self.grid_cb, SIGNAL('stateChanged(int)'), self.on_draw)
        
        #slider_label = QLabel('Bar width (%):')
        #self.slider = QSlider(Qt.Horizontal)
        #self.slider.setRange(1, 100)
        #self.slider.setValue(20)
        #self.slider.setTracking(True)
        #self.slider.setTickPosition(QSlider.TicksBothSides)
        #self.connect(self.slider, SIGNAL('valueChanged(int)'), self.on_draw)
        
    
        # Layout with box sizers
        #hbox = QHBoxLayout()
        
        #for w in [  self.textbox, self.draw_button, self.grid_cb,
                    #slider_label, self.slider]:
            #hbox.addWidget(w)
            #hbox.setAlignment(w, Qt.AlignVCenter)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.tabWidget )
        #hbox.addWidget(buttonBox)
        hbox.addLayout(buttonBox)
        self.vbox.addLayout(hbox)
        
        
        self.main_frame.setLayout(self.vbox)
        self.setCentralWidget(self.main_frame)
    
    #====================================================================================================
    def add_button(self, func, label, colour):
            
        def colourtuple( colour, alpha=1, ncols=255 ):
            rgba01 = np.array(colorConverter.to_rgba( colour, alpha=alpha ))    #rgba array range [0,1]
            return tuple( (ncols*rgba01).astype(int) )
        
        bg_colour = colourtuple( colour )
        hover_colour = colourtuple( colour, alpha=.7 )
        press_colour = colourtuple( "blue", alpha=.7 )
        
        button = QPushButton(label, self)
        style = Template("""
            QPushButton
            { 
                background-color: rgba$bg_colour;
                border-style: outset;
                border-width: 1px;
                border-radius: 3px;
                border-color: black;
                font: bold 14px;
                min-width: 10em;
                padding: 6px;
            }
            QPushButton:hover { background-color: rgba$hover_colour }
            QPushButton:pressed { background-color: rgba$press_colour }
            """).substitute( bg_colour=bg_colour, hover_colour=hover_colour,  press_colour=press_colour )
        
        button.setStyleSheet( style ) 
        button.clicked.connect( func )
        
        return button
        
    #====================================================================================================
    def create_buttons(self):
        ''' Add axes Buttons '''
        from string import Template
        
        buttonBox = QVBoxLayout()
         
        self.buttons = {}
        labels = ['load coords', 'daofind', 'phot', 'Ap. Corr.']
        func_names = ['load_coo', '_on_find_button', '_on_phot_button',  '_on_apcor_button']
        colours = ['g','g', 'orange', 'orange']
        for i, label in enumerate(labels):
            #F = getattr( self,  func_names[i] )
            F = self._on_click#lambda s : print(s)
            button = self.add_button(F, labels[i], colours[i])
            self.buttons[labels[i]] = button
            buttonBox.addWidget( button )
        
        buttonBox.addItem( QSpacerItem(20,200) )
        
        return buttonBox
    
    #====================================================================================================
    def _on_click(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' was pressed')
    
    #====================================================================================================
    def load_coo( self, fn ):
        '''load star cordinates from file and plot on image.'''
   
        #ffn, phfn = self.filenames.found_coo, self.filenames.phot_coo
        #for fn in (ffn, phfn):
        if os.path.isfile( fn ):
            
            print( 'Loading coordinate data from {}.'.format( fn ) )
            data = np.loadtxt(ffn, unpack=0)
            
            if data.shape[-1]==7:                         #IRAF generated coordinate files. TODO: files generated by this script should match this format
                coords, ids = data[:,:2], data[:,6]
            elif data.shape[-1]==2:
                coords = data
                ids = range(data.shape[0])
            
            elif 0 in data.shape:
                warn( 'Coordinate file "{}" contains no data!'.format(fn) )
                return
            else:
                warn( 'Coordinate file "{}" format not understood.'.format(fn) )
            
            if fn==ffn:
                coords -= 1    #Iraf uses 1-base indexing??
            #break
            
        else:
            print( 'No coordinate file exists!' )
            return
    
        return coords, ids
        
    #====================================================================================================
    #def
    
    
    

def main():
    app = QApplication(sys.argv)
    ui = StarSelector()
    ui.show()
    #from IPython import embed
    #embed()
    sys.exit( app.exec_() )


if __name__ == "__main__":
    main()
