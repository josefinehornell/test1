import numpy as np
import scipy.linalg as splg
import scipy.sparse as spsp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

mass = 1
omega = 1
Xmax = 8
h = 0.1
x = np.arange(-Xmax, Xmax+h, h).reshape(-1, 1)

m = len(x)

D22 =1*np.diag(np.ones(m-1),1) -  2*np.diag(np.ones(m),0) + 1*np.diag(np.ones(m-1),-1)
D24 = -1/12*np.diag(np.ones(m-2),2) + 4/3*np.diag(np.ones(m-1),1) \
        - 5/2*np.diag(np.ones(m),0)\
        + 4/3*np.diag(np.ones(m-1),-1)-1/12*np.diag(np.ones(m-2),-2)
D28 =   -1/560*np.diag(np.ones(m-4),4) + 8/315*np.diag(np.ones(m-3),3) \
        - 1/5*np.diag(np.ones(m-2),2) +8/5*np.diag(np.ones(m-1),1)\
        -205/72*np.diag(np.ones(m),0)\
        -1/560*np.diag(np.ones(m-4),-4) + 8/315*np.diag(np.ones(m-3),-3) \
        -1/5*np.diag(np.ones(m-2),-2) + 8/5*np.diag(np.ones(m-1),-1) 

D22 = spsp.csc_matrix(D22)
D24 = spsp.csc_matrix(D24)
D28 = spsp.csc_matrix(D28)

#Kinetic energy 
T = D28
T = -1.0/(2.0*mass*h*h)*T;

V = 0.5*mass*omega**2*x**2*np.diag(np.ones(m),0);
#Jag vill göra om V till sparse men då funkar inte eigsh
#V = spsp.csc_matrix(V)
H = T + V
eigvals, eigvecs = eigsh(H, k=m)

# Print the eigenvalues
print("Eigenvalues:")
print(eigvals[:5])


plt.figure(1)
plt.plot(x,0.5*mass*omega**2*x**2)
plt.title('Harmonisk oscillator Potential')
plt.xlabel('r')
plt.ylabel('V(r) potential')
plt.figure(2)
plt.title('Egenenergier')
plt.xlabel('r')
plt.ylabel('Energi')
plt.plot(x, eigvecs[:, 0])
plt.show()
