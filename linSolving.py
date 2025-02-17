import numpy as np
import scipy
import scipy.sparse as sparse
import sys

class Method:
  
    def bases(self, A, b):    
      self.A = A
      self.b = b

    def __init__(self,A,b):
      if(np.linalg.norm(b)==0):
         sys.exit("Errore: 'b' è il vettore nullo!")
      else:
         scipy.linalg.cholesky(A.todense())
      self.bases(A,b)


    def solveSys(self, sol, tol, maxIter):
       [k,res,control_tol,sol]=self.setupSolver(sol,tol)
       while (np.linalg.norm(res) >= control_tol) and (k < maxIter):
          sol += self.update(res)
          res = self.b - self.A*sol
          k += 1
       ris=[sol,k]
       return ris
    
    def setupSolver(self,sol,tol):
       k=0
       res=self.b-self.A*sol
       control_tol=tol*np.linalg.norm(self.b)
       setup=[k,res,control_tol,sol]
       return setup
    
    def update(self,res):
      pass
   
class Jacobi(Method):
    
    def __init__(self, A, b):
      super().__init__(A,b)
      self.P = sparse.diags(1/A.diagonal(),format="csr")
      
    def update(self,res):
        ris = self.P*res
        return ris
    
class Seidel(Method):

  def __init__(self, A, b):
    super().__init__(A,b)
    self.P=sparse.tril(A,format="csr")
  
  def update(self,res):   
    ris = scipy.sparse.linalg.spsolve_triangular(self.P,res,lower= True)
    return ris
  
class Gradiente(Method):

  def __init__(self, A, b):
    super().__init__(A,b)
    self.bases(A, b) 
  
  def update(self,res):
    y = self.A*res
    alfa = np.dot(res,res)
    beta = np.dot(res,y)
    ris = (alfa/beta)*res
    return ris

class Coniugato(Method):

  def __init__(self, A, b):
    super().__init__(A,b)
    self.bases(A, b) 
    self.d=np.zeros(len(b))
  
  def setupSolver(self,sol,tol):
       self.d=self.b-self.A*sol
       iter1=Gradiente(self.A,self.b)
       [sol,k]=iter1.solveSys(sol,tol,1)
       res=self.b-self.A*sol
       control_tol=tol*np.linalg.norm(self.b)
       setup=[k,res,control_tol,sol]
       return setup
  
  def update(self,res):
          self.setDir(res)
          y = self.A*self.d
          alfa = (np.dot(self.d,res))/(np.dot(self.d,y))
          ris = alfa*self.d
          return ris

  def setDir(self,res):
          w = self.A*res
          y = self.A*self.d
          beta = (np.dot(self.d,w))/(np.dot(self.d,y))
          self.d = res - beta*self.d