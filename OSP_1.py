import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import signal
import random

def Make_signal(s):
    t=np.arange(0,1,0.01)*2*np.pi  #初始化時間並轉成角度顯示(2p=360度)
    sin=1+np.sin(t) #製造正弦波 make sin wave
    square=1+signal.square(t) #製造方波 make square wave
    triangle=signal.sawtooth(2*np.pi*t) #製造三角波 make triangle
    mixA=sin+square #將正弦波及方波混和 mix sin and square
    mixB=sin+square+triangle #將正弦波 方波 及 三角波 混和 mix sin square and triangle mix
    plt.subplot(221)#============================================= show the mix1
    plt.title('mixA')
    plt.plot(t,mixA)
    plt.subplot(222)
    plt.title('mixB')
    plt.plot(t,mixB)#============================================= show the mix1

    u=np.zeros([100,2],float) #
    d=np.zeros([100,1],float)
    for i in range(100):
        u[i][0]=sin[i]
        u[i][1]=square[i]
        d[i][0]=triangle[i]

    return u,d,np.array(mixA),np.array(mixB)


def Make_PuT(u):

    uT=np.transpose(u)
    uTu=uT.dot(u)
    uTu_inverce=inv(uTu)
    # print(uTu_inverce)
    uTu_inverce_uT=uTu_inverce.dot(uT)
    id_matrix=np.identity(100)
    u_uTu_invers_uT=u.dot(uTu_inverce_uT)
    PuT=id_matrix-u_uTu_invers_uT    #identity matrix - pseudo_inverse
    return PuT


def OSP(d,u,x):

    PuT=Make_PuT(u)
    PuTR=np.transpose(x.dot(PuT)) # R the spectral (PuT*R)
    PuTR_TR=np.transpose(d) # transpose of D
    OSP_result=PuTR_TR.dot(PuTR) #the result of mix3
    return OSP_result

if __name__=='__main__':

    sample_signal=100
    u , d , mixA , mixB = Make_signal(sample_signal) #call our every signal

    sample_number=10                                #make 10 random case to detect
    input_data=[mixA,mixB]
    dict={0:'mixA',1:'mixB'}
    seed = [random.randint(0,1) for x in range(sample_number)]
    test_space=[dict[i] for i in seed]
    print(test_space)
    testcase=[input_data[x] for x in seed]
    ans=[OSP(d,u,x) for x in testcase]
    plt.subplot(223)
    plt.plot(ans)
    plt.show()




