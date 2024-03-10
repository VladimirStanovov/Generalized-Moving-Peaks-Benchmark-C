import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-100, 100, 101)
y = np.linspace(-100, 100, 101)
for env in range(1,100):
    filename = f"res_e{env}.txt"
    X, Y = np.meshgrid(x, y)
    Z = np.loadtxt(filename)
    print(env,np.min(Z),np.max(Z))
    #print(Z)
    fig = plt.figure(figsize=(10,10))
    plt.contour(X,Y,Z,50,cmap="jet")
    fig.savefig(f"env_{env}.png")    
    plt.close(fig)
