import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtGui

import vlc

class ExWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        p = vlc.Instance()
        player = p.media_player_new()
        Media = p.media_new('ani01_Hi_Short.mov')
        Media.get_mrl()
        player.set_media(Media)

        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Main Window')
        self.show()

        player.play()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExWindow()
    sys.exit(app.exec_())


'''
import vlc
import sys

if sys.platform == "darwin":
    from PyQt4 import QtCore
    from PyQt4 import QtGui

    p = vlc.Instance()
    player = p.media_player_new()
    Media = p.media_new('ani01_Hi_Short.mov')
    Media.get_mrl()
    player.set_media(Media)

    vlcApp = QtGui.QApplication(sys.argv)
    vlcWidget = QtGui.QFrame()
    vlcWidget.resize(700, 700)
    vlcWidget.show()
    player.set_nsobject(vlcWidget.winId())
    
'''

#p=vlc.MediaPlayer('./ani01_Hi_Short.mov')
#p.play()
#p.get_instance() # returns the corresponding instance
