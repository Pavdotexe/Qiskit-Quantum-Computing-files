import numpy as np 
import math 

n=4 #Qubits being simulated
def psi(string_num):
    dec=bin2dec(string_num)
    return dec2vec(dec,len(string_num))

def bin2dec(string_num):
    return int(string_num, 2)

def dec2vec(dec,n):
    vec=np.zeros((2**n,1))
    vec[dec,0]=1
    return vec

def hadamard(n):
    """
    Creates a Hadamard Matrix
    """
    r2=math.sqrt(2.0)
    H1=np.array([[1/r2,1/r2],[1/r2,-1/r2]])
    if n==1:
        H=H1
    else:
        H=1
        for i in range(1,n+1):
            H=np.kron(H,H1) #kron product of matrices 
    return H

def U_Oracle(n):
    """
    Oracle for marking highest fitness to use in Grover's Algorithm
    """
    zero_mat=np.zeros((2**n,2**n))
    for i in range(0,2**n): #defining the oracle with the following rule
        if i==5:
            O=1
        else:
            O=0
    # Inverter
        zero_mat[i,i]=(-1)**O
    return zero_mat

def invert_about_average(n):
    """
    Inverting about new average
    """
    ia_mat=2*np.ones(2**n)/(2**n)
    ia_mat=ia_mat-np.identity(2**n)
    return ia_mat

def maxiter(n):
    """
    Grover's Maximum Iterations 
    """
    max_iter=(np.pi/4)*math.sqrt(2**n)
    return max_iter
def grover(n,string_num):
    psi_=psi(string_num)
    H=hadamard(n)
    psi_=np.dot(H,psi_) #putting it into superposition
    print(psi_)
    print()
    iter=np.trunc(maxiter(n))
    iter=int(round(iter))
    for i in range (1,iter):
        U_O=U_Oracle(n)
        print(U_O)
        print()
        psi_=np.dot(U_O,psi_)
        print(psi_)
        print()
        D=invert_about_average(n)
        psi_=np.dot(D,psi_)
    print(psi_)

grover(4,'0000')
