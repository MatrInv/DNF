import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from random import uniform

def euclidean_distance(u, v):
        return math.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2)

class Gaussian:
    
    def __init__(self, sigma, c):
        self.sigma = sigma
        self.c = c
    
    def activity(self, z):
        return self.c*math.exp(-(z**2)/(2*(self.sigma**2)))
        
        
class NeuralField :
    
    def __init__(self, w, h, dt, tau, gaussEx, gaussIn):
        self.W = w
        self.H = h
        self.field = [[0 for _ in range(w)] for _ in range(h)]
        self.dt = dt
        self.time = 0.0
        self.input = [[0 for _ in range(w)] for _ in range(h)]
        self.excitation = gaussEx
        self.inhibition = gaussIn
        self.tau = tau
       
    def euclidean_norm(self, u, v):
        return math.sqrt( ( ((u[0]-v[0])/self.W)**2 + ((u[1]-v[1])/self.H)**2 ) / 2 )
        
    def u(self, x):
        return self.field[x[1]][x[0]]
    
    def set_u(self, x, val):
        self.field[x[1]][x[0]] = val
        
    #difference of gaussians
    def w(self, z):
        return self.excitation.activity(z) - self.inhibition.activity(z)
    
    def input_activity(self, x):
        return self.input[x[1]][x[0]]
    
    def set_input(self, inp):
        self.input = inp
        
    def u_norm(self, val):
        if val > 1:
            return 1
        if val < 0:
            return 0
        return val
        
    def local_update(self, x):
        lateralInteractions = 0.0
        for j in range(self.H) :
            for i in range(self.W) :
                lateralInteractions = lateralInteractions + self.u([i,j])*self.w(self.euclidean_norm(x, [i, j]))    
        res = self.u(x) + self.dt * ( -self.u(x) + lateralInteractions + self.input_activity(x) ) / self.tau
        return self.u_norm(res)
        
    def global_update_sync(self):
        buffer = [[0 for _ in range(self.W)] for _ in range(self.H)]
        for j in range(self.H):
            for i in range(self.W):
                buffer[j][i] = self.local_update([i,j])
        self.field = buffer
        
    def global_update_async(self):
        print("async update : TO DO")
        
    def update_input(self):
         #creating input 'activ'
        gaussActiv = Gaussian(2,2)
        #gaussActiv2 = Gaussian(2,2)
        activ = [[0.0 for _ in range(self.W)] for _ in range(self.H)]
        for j in range(self.H) :
            for i in range(self.W) :
                activ[j][i] = activ[j][i] + gaussActiv.activity(euclidean_distance([math.cos(self.time)*10+15,10], [i,j])) #+ gaussActiv2.activity(euclidean_distance([20,20], [i,j]))
        self.set_input(activ)
        
def main():
    #parameters initialization
    H = 30
    W = 30
    sigmaEx = 12.0
    sigmaIn = 10.0
    coefIn = 0.7
    coefEx = 0.68
    dt = 0.1
    tau = 0.64
    
    exitaG = Gaussian(sigmaEx, coefEx) 
    inhibG = Gaussian(sigmaIn, coefIn)
    
    dnf = NeuralField(W, H, dt, tau, exitaG, inhibG)

    #ploting
    fig = plt.figure("Tracking")
    sp1 = plt.subplot(1,2,1)
    b = np.random.random((W, H))
    imIn = plt.imshow(b, cmap='hot', interpolation='nearest', animated=True)
    sp1.set_title('Input')
    a = np.random.random((W, H))
    sp2 = plt.subplot(1,2,2)
    im = plt.imshow(a, cmap='hot', interpolation='nearest', animated=True)
    sp2.set_title('Dynamic Neural Field (sync)')
    
    def update(*args):
        dnf.update_input()
        imIn.set_array(dnf.input)
        
        dnf.global_update_sync()
        im.set_array(dnf.field)
        
        dnf.time=dnf.time+dt
        return im, imIn,
    
    ani = animation.FuncAnimation(fig, update, interval=30, blit=True)
    plt.show()
    
if __name__ == '__main__':
    main()
