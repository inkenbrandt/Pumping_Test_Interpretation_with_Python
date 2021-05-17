### main script ###
from functools import partial

from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, uic, QtGui

import pandas as pd
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pumpingtest import *
# support for user interfacce

#https://stackoverflow.com/questions/48603517/how-to-get-qtableview-right-clicked-index

class LoginPage(QtWidgets.QDialog):
    def __init__(self, parent, wellcols, *args, **kwargs):
        #https: // stackoverflow.com / questions / 48247643 / how - to - dynamic - update - qcombobox - in -pyqt5
        super(LoginPage, self).__init__(parent, *args, **kwargs)
        uic.loadUi('add_well_data_dialog.ui', self)
        self.welln = None

        for i in wellcols:
            self.welluploadcombo.addItem(str(i))

        self.buttonBox.accepted.connect(self.acceptbutt)
        self.buttonBox.rejected.connect(self.reject)

    def acceptbutt(self):
        #self.completed = 0
        print('yes')
        self.welln = str(self.welluploadcombo.currentText())
        self.accept()

    def getfile(self):
        self.name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Save File')
        print(self.name)



class TableModel(QtCore.QAbstractTableModel):
    #https://stackoverflow.com/questions/37786299/how-to-delete-row-rows-from-a-qtableview-in-pyqt
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == QtCore.Qt.Vertical:
                return str(self._data.index[section])

    def flags(self, index):
        """
        Make table editable.
        make first column non editable
        :param index:
        :return:
        """
        if index.column() > -1:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
        else:
            return QtCore.Qt.ItemIsSelectable


    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.EditRole:
            row = index.row()
            col = index.column()
            try:
                self._data.iloc[row][col] = value
            except:
                return False
            return True

    def insertRows(self):
        row_count = len(self._data)
        self.beginInsertRows(QtCore.QModelIndex(), row_count, row_count)
        empty_data = {key: None for key in self._data.columns if not key=='_id'}

        self._data = self._data.append(empty_data, ignore_index=True)
        row_count += 1
        self.endInsertRows()
        return True

    def removeDataFrameRows(self, rows):
        print('yes')
        position = min(rows)
        count = len(rows)
        print(count)
        self.beginRemoveRows(QtCore.QModelIndex(), position, position + count - 1)

        removedAny = False
        for idx, line in self._data.iterrows():
            if idx in rows:
                removedAny = True
                self._data.drop(idx, inplace=True)

        if not removedAny:
            return False

        self._data.reset_index(inplace=True, drop=True)

        self.endRemoveRows()
        return True


class GUI(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(GUI, self).__init__(*args, **kwargs)
        # initiate GUI

        qtCreatorFile = 'pumping_test_interface.ui'
        uic.loadUi(qtCreatorFile,self)

        self.StationModel = {}

        aquifer = Aquifer()

        # Setup Diagram for plotting
        self.dpi = 100
        self.fig = Figure((4.5, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.aquifertestframe)
        self.axes = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas, self.aquifertestframe)

        # load values read from files
        self.KhInput.setText(str(aquifer.K))
        self.SsInput.setText(str(aquifer.Ss))
        self.SyInput.setText(str(aquifer.Sy))
        self.bInput.setText(str(aquifer.b))
        self.bcInput.setText(str(aquifer.bc))
        self.KcInput.setText(str(aquifer.Kc))
        self.SscInput.setText(str(aquifer.Ssc))


        welldf = pd.DataFrame(columns=['Well Name',
                                       'X','Y',
                                       'radius',
                                       'depth',
                                       'pumped aquifer',
                                       'Pump File','Obs File'])
        self.WellsModel = TableModel(welldf)
        #self.StationModel = TableModel(df)
        self.welltableview.setModel(self.WellsModel)

        self.tabWidget.currentChanged.connect(self.printtab)

        # button functionality
        self.add_well_button.clicked.connect(self.WellsModel.insertRows)
        self.remove_well_button.clicked.connect(partial(self.delete_record1, self.WellsModel, self.welltableview))
        self.add_well_record.clicked.connect(self.executeLoginPage) #self.add_well_record.clicked.connect(self.openFileNameDialog) #

        self.pushUpdate.clicked.connect(partial(self.Update, aquifer))
        self.pushEval.clicked.connect(partial(self.Evaluate, aquifer))

        self.pushSave.clicked.connect(partial(self.SaveFiles, aquifer))
        self.pushExit.clicked.connect(self.close)

        self.wellnamescombo2.activated.connect(self.newview)
        #self.refreshbutton.clicked.connect(self.newview)

    def newview(self):
        print(str(self.wellnamescombo2.currentText()))
        self.welln = str(self.wellnamescombo2.currentText())
        print(self.welln)
        if self.welln in self.StationModel.keys():
            self.StationTableView.setModel(self.StationModel[self.welln])

    def printtab(self):
        if self.tabWidget.currentIndex() == 2:
            self.wellnamescombo2.clear()
            for i in self.StationModel.keys():
                self.wellnamescombo2.addItem(str(i))

    def executeLoginPage(self, s):
        wellcols = self.WellsModel._data['Well Name'].values
        self.dlg = LoginPage(self, wellcols)
        if self.dlg.exec_():
            print("Success!")
            self.welln = self.dlg.welln
            print(self.welln)
            self.openFileNameDialog()
            #self.dlg.chemdata
            #self.dfd = self.dlg.userpw
            #self.add_data()
        else:
            print("Cancel!")

    def delete_record1(self, model, view):
        """Delete rows with currently selected cells and/or selected rows of the model"""
        rows = [model_index.row() for model_index in view.selectedIndexes()]
        rows = list(set(rows))
        print(rows)
        #rows.sort(reverse=True)
        #for i in rows:
            #model.removeRow(i)
            #model.removeRow(i)
        model.removeDataFrameRows(rows)
        #self._data = self._data.drop(self._data.iloc[rows], axis=0)
        #model.submitAll()
        #model.select()


    def Update(self, aquifer):
        # update aquifer and well objects with values on form
        aquifer.K = float(self.KhInput.text())
        aquifer.Ss = float(self.SsInput.text())
        aquifer.Sy = float(self.SyInput.text())
        aquifer.b = float(self.bInput.text())
        aquifer.bc = float(self.bcInput.text())
        aquifer.Kc = float(self.KcInput.text())
        aquifer.Ssc = float(self.SscInput.text())
        aquifer.S = aquifer.Ss * aquifer.b
        self.well.r = float(self.rInput.text())
        self.well.Q = float(self.QInput.text())

    def SaveFiles(self, aquifer):
        # write current model to aquifer and well files
        aquifer.WriteValues()
        self.well.WriteValues()

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if self.fileName:
            print(self.fileName)
            df = pd.read_csv(self.fileName, sep="\t")
            self.StationModel[self.welln] = TableModel(df)
            self.StationTableView.setModel(self.StationModel[self.welln])
            print(self.StationModel[self.welln]._data)
            #self.data = DataSet(df, t='t (day)', s='s')
            self.well = Well(self.StationModel[self.welln]._data['t (day)'].min(),
                             self.StationModel[self.welln]._data['t (day)'].max())
            self.rInput.setText(str(self.well.r))
            self.QInput.setText(str(self.well.Q))
            self.wellnamescombo2.clear()
            for i in self.StationModel.keys():
                self.wellnamescombo2.addItem(str(i))

    def Evaluate(self, aquifer):

        self.axes.cla()

        # set up test objects using current parameter values
        theis = Theis(aquifer, self.well)
        hantush = Hantush(aquifer, self.well)
        shortStor = ShortStorage(aquifer, self.well)
        numericWaterTable = MOL(aquifer, self.well)

        # plot transducer data
        self.axes.scatter(self.StationModel[self.welln]._data['t (day)'],
                          self.StationModel[self.welln]._data['s'],
                          s=10, facecolors='none', edgecolors='black', label='Data')

        # run checked models and add to plot
        if self.checkTheisConf.checkState():
            sTheisC = theis.Drawdown(0)
            self.axes.plot(self.well.tArray, sTheisC, color='red', label='Confined (Theis)')
        if self.checkMOLTheis.checkState():
            sMOLt = numericWaterTable.Drawdown(1)
            self.axes.plot(self.well.tArray, sMOLt, color='magenta', label='Confined (wellbore storage)')
        if self.checkHantush.checkState():
            sHantush = hantush.Drawdown()
            self.axes.plot(self.well.tArray, sHantush, color='green', label='Leaky (Hantush & Jacob)')
        if self.checkShortStor.checkState():
            sShortStor = shortStor.Drawdown()
            self.axes.plot(self.well.tArray, sShortStor, color='olive', label='Leaky (Hantush, 1960)')
        if self.checkTheisUnconf.checkState():
            sTheisU = theis.Drawdown(1)
            self.axes.plot(self.well.tArray, sTheisU, color='blue', label='Unconfined (Theis, with Sy)')
        if self.checkMOLDupuit.checkState():
            sMOLd = numericWaterTable.Drawdown(0)
            self.axes.plot(self.well.tArray, sMOLd, color='cyan', label='Unconfined (Dupuit; numerical)')

        self.axes.set_xscale('log')
        self.axes.set_yscale('log')
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Drawdown')
        self.axes.legend(loc=4)

        self.canvas.draw()

    def changeunits(self):
        """make all input units consistent for analysis"""

def PumpTest():
    # read parameters

    # set up GUI
    app = QtCore.QCoreApplication.instance()
    if app is None: app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())


# run script
PumpTest()