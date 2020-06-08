import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidget, QTableWidgetItem
from PyQt5 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve
from math import e

class programa(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("programa.ui", self)
        #botones

        self.graficar.clicked.connect(self.grafica)
        self.aceptar.clicked.connect(self.cantidad)
        self.lineal.clicked.connect(self.reglineal)
        self.slg.clicked.connect(self.regresionslg)
        self.pol.clicked.connect(self.regresionpol)
        self.log.clicked.connect(self.regresionlg)
    def cantidad(self):
        canti = self.linea.text()
        canti=int(canti)
        self.tableWidget.setRowCount(canti)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(('Datos x', 'Datos y'))
        return canti
    def datos(self):

        lista=[]
        pro=[]
        for x in range(self.cantidad()):

            elementox=self.tableWidget.item(x, 0).text()
            elementox=float(elementox)
            pro.append(elementox)

            elementoy=self.tableWidget.item(x, 1).text()
            elementoy=float(elementoy)
            pro.append(elementoy)

            lista.append(pro)
            pro=[]

        return lista

    def grafica(self):

        data = np.array(self.datos())

        variable1 = np.array(data[:, 0])
        variable2 = np.array(data[:, 1])

        graph_title = "grafica 1"

        graph_x = "x"

        graph_y = "y"

        plt.plot(variable1, variable2, 'b.')
        plt.suptitle(graph_title)
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()

    def reglineal(self):
        datos = np.array(self.datos())
        slope, intercept, r_value, p_value, std_err = stats.linregress(datos[:, 0], datos[:, 1])
        xt = np.linspace(0, datos[-1, -1] + 0.5, 1000)
        rl = xt * slope + intercept
        st = ' Ec. encontrada : y= {}x + {}'.format(slope, intercept)
        self.reglin.setText(st)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out) = (xt, rl, st)
        regraph = True
        graph_title = "Regresion Lineal"

        graph_x = "X"

        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')
        if regraph:
            plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()

    def regresionlg(self):
        datos = np.array(self.datos())
        ly = np.log(datos[:, 1])
        n = np.size(datos[:, 0])
        lx = np.log(datos[:, 0])
        ly_mul_lx = ly * lx
        lx_exp2 = lx ** 2

        def equations(p):
            a, b = p
            return (n * np.log(a) + b * np.sum(lx) - np.sum(ly),
                    np.log(a) * np.sum(lx) + b * np.sum(lx_exp2) - np.sum(ly_mul_lx))

        a, b = fsolve(equations, (1, 1))

        xt = np.linspace(0, datos[-1, -2] + 0.5, 1000)
        rlg = a * (xt) ** b

        out = 'Ec. encontrada : {}x^{}'.format(a, b)
        self.reglg.setText(out)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out)= (xt, rlg, out)
        graph_title = "Regresion Semi-log"

        graph_x = "X"

        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')

        plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
    def regresionslg(self):
        datos = np.array(self.datos())
        ly = np.log(datos[:, 1])
        x_exp2 = np.square(datos[:, 0])
        x_lny = datos[:, 0] * np.log(datos[:, 1])
        mean_x = np.mean(datos[:, 0])
        mean_ly = np.mean(ly)

        m = ((np.sum(x_lny) - mean_ly * np.sum(datos[:, 0])) / (np.sum(x_exp2) - mean_x * np.sum(datos[:, 0])))

        b = e ** (mean_ly - m * mean_x)

        xt = np.linspace(0, datos[-1, -1] + 0.5, 1000)
        rslg = b * (e) ** (m * xt)
        out = 'Ec. encontrada : {}e^({}x)'.format(b, m)
        self.regslg.setText(out)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out) = (xt, rslg, out)

        graph_title = "Regresion Semi-log"

        graph_x = "X"

        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')

        plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
    def regresionpol(self):
        datos = np.array(self.datos())
        g = int(self.grad.text())
        n = np.size(datos[:, 0])

        A = np.empty([g + 1, g + 1])
        A[0, 0] = n

        for k in range(1, g + 1, 1):
            t = np.sum(datos[:, 0] ** k)
            i = 0
            j = k
            while (j >= 0 and j <= g and i <= g):
                A[i, j] = t
                i += 1
                j -= 1

        for k in range(g + 1, g * 2 + 1, 1):
            t = np.sum(datos[:, 0] ** k)
            i = g
            j = k - g
            while (j <= g and i <= g):
                A[i, j] = t
                i -= 1
                j += 1

        B = np.empty((g + 1))

        for i in range(0, g + 1):
            l = np.sum(datos[:, 1] * (datos[:, 0] ** i))
            B[i] = l

        sol = np.linalg.solve(A, B)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])


        def PolyCoefficients(x, coeffs):
            """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.
            The coefficients must be in ascending order (``x**0`` to ``x**o``).
            """

            o = len(coeffs)

            z = 0
            for i in range(o):
                z += coeffs[i] * x ** i
            return z

        xt = np.linspace(0, datos[-1, -2] + 0.5, 1000)

        p = PolyCoefficients(xt, sol)

        out = 'Coeficientes del polinomio, orden ascendente: {}'.format(sol)
        self.cua.setText(out)
        (xe, ye, out)=(xt, p, out)
        graph_title = "Regresion Polinomial"

        graph_x = "X"

        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')

        plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
if __name__ == "__main__":

    prog = QApplication(sys.argv)
    GUI = programa()
    GUI.show()
    sys.exit(prog.exec_())

