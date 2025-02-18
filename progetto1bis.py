
import linSolving
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# setting variables to solve system Ax=b 
A = sio.mmread("dati/spa1.mtx").tocsr()
x = np.ones(A.get_shape()[1])
b = A*x
# setting thresholds for the methods to stop
tol = [1.e-04, 1.e-06, 1.e-08, 1.e-10]
maxIter = 3.e+4
# start comparison between these four methods
metodo = [linSolving.Jacobi(A,b), linSolving.Gradiente(A,b), linSolving.Coniugato(A,b), linSolving.Seidel(A,b)]

for j in range(0,len(metodo)):
 print("Metodo: " + str(metodo[j].__class__.__name__))
 for i in range(0,len(tol)):
    x0 = np.zeros(len(x))
    comp_time = time.time()
    results = metodo[j].solveSys(x0,tol[i],maxIter)
    comp_time = time.time() - comp_time
    er = np.linalg.norm(x - results[0])/np.linalg.norm(x)
    print("Tolerance = " + str(tol[i]))
    print("Errore relativo = " + str(er))
    print("Numero iterazioni = " + str(results[1]))
    print("Tempo di calcolo = " + str(comp_time))






