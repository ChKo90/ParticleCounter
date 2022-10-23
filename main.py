
import sys
from PyQt5 import QtWidgets

from gui.gui import Ui

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    if 'get_ipython' in globals():
        window.show()
    else:
        sys.exit(app.exec_()) # Start the application