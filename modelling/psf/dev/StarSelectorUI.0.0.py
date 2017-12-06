from mplMultiTab import *
from PyQt4.QtGui import QPushButton, QHBoxLayout
from PyQt4 import QtCore

from matplotlib.colors import colorConverter
from string import Template

class StarSelector(MplMultiTab):

    #====================================================================================================
    def create_main_frame(self, figures, labels):
        
        MplMultiTab.create_main_frame( self, figures, labels )
        
        #####  Add axes Buttons #####
        def create_button(func, label, colour):
            #ax = plt.axes(rect)
            #button = Button(ax, label)
            rgba = np.array(colorConverter.to_rgba( 'green' ))*255
            hovercolor = tuple( rgba.astype(int) )
            #ax.patch.set_fc( lightcolour )                             #set button facecolor
            #button.color = lightcolour
            button.hovercolor = colour
            button.status = 0
            button.on_clicked( func )
            return button
        
        pushButton = QPushButton('FOO!!', self)
        
        Template("
                QPushButton
                { 
                    background-color: $bg_colour;
                    border-style: outset;
                    border-width: 1px;
                    border-radius: 3px;
                    border-color: black;
                    font: bold 14px;
                    min-width: 10em;
                    padding: 6px;
                }
                QPushButton:hover { background-color: $hover_colour }
                QPushButton:pressed { background-color: $press_colour }
                ").substitute( bg_colour=bg_colour, hover_colour=hover_colour,  press_colour=press_colour )
        
        print
        
        pushButton.setStyleSheet( )#rgb(255, 255,255) border:3px solid rgb(255, 170, 255); 
        
        pushButton.clicked.connect( self._on_click )
        
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
        hbox.addWidget(pushButton)
        self.vbox.addLayout(hbox)
        
        self.main_frame.setLayout(self.vbox)
        self.setCentralWidget(self.main_frame)
    
    #====================================================================================================
    def add_buttons(self):
        
        
        
        self.buttons = {}
        rect = np.array([0.88, 0.82, 0.1, 0.075])
        labels = ['load coords', 'daofind', 'phot', 'Ap. Corr.']
        func_names = ['load_coo', '_on_find_button', '_on_phot_button',  '_on_apcor_button']
        colours = ['g','g', 'orange', 'orange']
        for i, label in enumerate(labels):
            F = getattr( self,  func_names[i] )
            self.buttons[labels[i]] = create_button(rect, F, labels[i], colours[i])
            
            rect[1] -= 0.1              #move the y position of the next button down
    
    #====================================================================================================
    def _on_click(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' was pressed')
        
        

def main():
    app = QApplication(sys.argv)
    ui = StarSelector()
    ui.show()
    #from IPython import embed
    #embed()
    sys.exit( app.exec_() )


if __name__ == "__main__":
    main()
