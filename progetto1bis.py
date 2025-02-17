
import linSolving
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

A = sio.mmread("dati/spa1.mtx").tocsr()
# A = sio.mmread("dati/spa2.mtx").tocsr()
# A = sio.mmread("dati/vem1.mtx").tocsr() #toeplitz/poisson/band?
# A = sio.mmread("dati/vem2.mtx").tocsr() #toeplitz/poisson/band?

# print(A.getnnz()/(A.get_shape()[1]**2))
# print(A.get_shape())
# print(np.linalg.cond(A.todense()))
# plt.spy(A)
# plt.show()

# def is_diagonal(A):
#     A = np.abs(np.asarray(A))  # converts lists to numpy arrays
#     diagonal = np.diag(A)  # [7, 7, 7]
#     others = A - np.diag(diagonal)  # matrix 'A' but with zeros along the diagonal
#     return np.all(diagonal >= others.sum(axis=1))
# print(is_diagonal(A.todense()))

x = np.ones(A.get_shape()[1])
b = A*x
tol = [1.e-04, 1.e-06, 1.e-08, 1.e-10]
metodo = [linSolving.Jacobi(A,b), linSolving.Gradiente(A,b), linSolving.Coniugato(A,b), linSolving.Seidel(A,b)]
maxIter = 3.e+4

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
    # print(er,results[1],comp_time)






