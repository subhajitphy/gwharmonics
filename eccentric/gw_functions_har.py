import numpy as np
import scipy.integrate as integrate
from scipy.special import jv
from numpy import sin, cos, sqrt,tan, pi, log10
import antenna_pattern as ap
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from constants import *

def waveform(m1,m2,e,n,i,f_obs,n_obs,r,t,key):
    
    M=m1+m2
    eta=m1*m2/(m1+m2)**2
    Mc=M*eta**(3/5)
    BJ=jv(n,n*e)
    DJ=1/2*(jv(n-1,n*e)-jv(n+1,n*e))
    Sn=-2/e*sqrt(1-e**2)*DJ+2*n/e**2*(1-e**2)**(3/2)*BJ
    Cn=-(2-e**2)/e**2*BJ+2*(1-e**2)/e*DJ
    f_prim=f_obs/n_obs
    omg=f_prim*2*pi
    df=6*(2*pi)**(2/3)*(2*pi/omg)**(-5/3)*(G*M)**(2/3)/(c**2*(1-e**2))
    Phi=df*t
    h0=4*(2*pi)**(-2/3)*G**(5/3)*c**(-4)*Mc**(5/3)/r*(2*pi/omg)**(-2/3)
    
    fn=n*f_prim
    
    SCn=(Sn-Cn)/2
    SCp=(Sn+Cn)/2
    
    hp=(-h0/2*(sin(i)**2*BJ*cos(n*omg*t)+(1+cos(i)**2)
              *(SCn*cos(n*omg*t+key*2*Phi)+SCp*cos(n*omg*t-key*2*Phi))))
    
    hx=-h0*cos(i)*(SCn*sin(n*omg*t+2*Phi*key)+SCp*sin(n*omg*t-2*Phi*key))
    
    return hp, hx, [fn-df,fn,fn+df]
    
    
    
    
def response(m1,m2,e,n,i,f_obs,n_obs,r,ti,tf
             ,key,gwra, gwdec, psrra, psrdec,
             psigw):
    
    if type(tf) is np.ndarray:
        return np.array([response(m1,m2,e,n,i,f_obs,n_obs,r,ti,tf1
             ,key,gwra, gwdec, psrra, psrdec,
             psigw) for tf1 in tf])
    
    sp=integrate.quad(lambda t1: waveform(m1,m2,e,n,i,f_obs,n_obs,r,t1,key)[0],ti,tf)[0]
    sx=integrate.quad(lambda t1: waveform(m1,m2,e,n,i,f_obs,n_obs,r,t1,key)[1],ti,tf)[0]
    cosmu, Fp, Fx = ap.antenna_pattern(gwra, gwdec, psrra, psrdec)
    

    c2psi = np.cos(2*psigw)
    s2psi = np.sin(2*psigw)
    Rpsi = np.array([[c2psi, -s2psi],
                     [s2psi, c2psi]])
    
    res = np.dot([Fp,Fx], np.dot(Rpsi, [sp,sx]))
    
    return res            