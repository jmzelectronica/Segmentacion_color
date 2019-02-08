
import numpy as np
import matplotlib.pyplot as plt  
def particleGenerator(dist,sigma,Nparticles,x0,y0):
    x = np.arange(-dist,dist,0.1)
    y = np.sqrt((dist**2)-x**2)
    rUp = dist+sigma*np.random.rand(Nparticles,1)
    rDown= dist-sigma*np.random.rand(Nparticles,1)
    theta = 2*np.pi*np.random.rand(Nparticles, 1)
    xup = x0 + rUp*np.cos(theta)
    yup = y0 + rUp*np.sin(theta)
    xdown = x0 + rDown*np.cos(theta)
    ydown = y0 + rDown*np.sin(theta)
    print (x,y,x,-y,xup,yup,xdown,ydown)
    plt.plot(x,y,'-g',x,-y,'-g',xup,yup,'*r', xdown,ydown,'*b')
    plt.show()
    pass


x = particleGenerator(50,10,100,2,2)
