"""
Copyright 2022 by Christian König.
All rights reserved.
"""

import sys
from PyQt5 import QtWidgets

from gui.gui import Ui

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    if 'get_ipython' in globals():
        window.show()
    else:
        sys.exit(app.exec_())