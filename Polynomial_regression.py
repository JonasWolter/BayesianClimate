# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:43:52 2019

@author: Jonas Laptop
"""

#Importiing required libraries
import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.close('all')
import warnings
warnings.filterwarnings('ignore')

#####################################################################
#Description: 
#This Program fits a basyesian polynomial regression of degree 3 with 
#to the temeprature on different locations in the world. This means that there are four parameters
# to be estimated. Those parameters are assumed to be normally distributed. In addition we work
# with a student noise model which gives an overall of 6 parameters to be estimated.
#
#The location (countries) can be chosen with the Location parameter. Three different approximation methods
#have been implemented wich can be chosen through the Method parameter. 
#Parameteres for different models are set in appropriate section. Different location may require 
#a change in the stepsize for MH or the scale for GVA.
#
#As the Output the script first gives the parameters of basic linear regression and the sum of least squares.
#Furthertmore it gives the estimates for a basic polynomial regression and the associated sum of squares. 
#Secondly approximates of the estimated parameters are given. These are either the estimates for the
#mean for the Gaussian for GVA and Laplace, or the mean value of the Markov chain after the burn-in period for 
#Metropolis Hasting. Furthermore the sum of squares of the estimate is printed. For MH the acceptance ratio is given as well.
#Lastly we see the runtime to get an idea of the performance of the used algortihm.
#In addition some graphs are plotted which should be self-explanatory.

Location='Global' #Selecting Location for which Data should be analysed, 'Global' for whole earth 
Method='MH'#Choose bettwen Laplace, Metropolis Hasting (MH) and Gaussian Variational Approximation (GVA) 
            #with a diagonal Hessian or GVA2 with the full Hessian. In general GVA2 is preferred but it 
            #takes longer to run. The default parameters yield acceptable results in a few minutes.
            #For more accurate results a longer time is required, which can be achieved by channging the parameters.  
    
#Note:GVA does not yield very good results for this model if we require a runtime of less than an hour.
#Explanations for this will be given in the presentation.

########################Function needed in code
def sum_handles3(handles_list):
    def aux(x1,x2,x3):
        temp=0
        for f in handles_list:
            temp=temp+f(x1,x2,x3)
        return temp
    return aux

def multi_Gamma(mu,S,a,b):
    d=(np.size(S))**0.5
    def mGaux(beta,Theta):
        return (b**a/sp.special.gamma(a))*(np.linalg.det(S)**(1/2))/((2*np.pi)**(d/2))* \
                np.exp(-b*beta-0.5*beta*np.transpose(Theta-mu)@S@(Theta-mu))
    return mGaux

#############################################Reading Data
if Location=='Global':
    StartYear=1830 #year from which temperatures are considered
    Datapoints=170#Number of datapoints considered in analysis
    # drop unnecessary columns
    global_temp = pd.read_csv('GlobalTemperatures.csv')
    global_temp = global_temp[['dt', 'LandAverageTemperature']]
    global_temp['dt'] = pd.to_datetime(global_temp['dt'])
    global_temp['year'] = global_temp['dt'].map(lambda x: x.year)
    global_temp['month'] = global_temp['dt'].map(lambda x: x.month)
    global_temp=global_temp[global_temp.year>StartYear]
    min_year = global_temp['year'].min()
    max_year = global_temp['year'].max()
    years = range(min_year, max_year + 1)
    total_temps=[]
    for year in years:
        curr_years_data = global_temp[global_temp['year'] == year]
        total_temps.append(curr_years_data.LandAverageTemperature.mean())
elif Location=='Switzerland':
    StartYear=1830
    Datapoints=180
    temp_countries=pd.read_csv('GlobalLandTemperaturesByCountry.csv')
    temp_Switzerland=temp_countries.loc[temp_countries.Country == 'Switzerland']
    temp_Switzerland = temp_Switzerland[['dt', 'AverageTemperature']]
    temp_Switzerland['dt'] = pd.to_datetime(temp_Switzerland['dt'])
    temp_Switzerland['year'] = temp_Switzerland['dt'].map(lambda x: x.year)
    temp_Switzerland['month'] = temp_Switzerland['dt'].map(lambda x: x.month)
    temp_Switzerland=temp_Switzerland[temp_Switzerland.year>StartYear]
    min_year = temp_Switzerland['year'].min()
    max_year = temp_Switzerland['year'].max()
    years = range(min_year, max_year + 1)
    total_temps=[]
    for year in years:
        curr_years_data = temp_Switzerland[temp_Switzerland['year'] == year]
        total_temps.append(curr_years_data.AverageTemperature.mean())
else:
    StartYear=1900
    Datapoints=110
    print('Country selected:', Location)
    temp_countries=pd.read_csv('GlobalLandTemperaturesByCountry.csv')
    temp_Conutry=temp_countries.loc[temp_countries.Country == Location]
    temp_Conutry = temp_Conutry[['dt', 'AverageTemperature']]
    temp_Conutry['dt'] = pd.to_datetime(temp_Conutry['dt'])
    temp_Conutry['year'] = temp_Conutry['dt'].map(lambda x: x.year)
    temp_Conutry['month'] = temp_Conutry['dt'].map(lambda x: x.month)
    temp_Conutry=temp_Conutry[temp_Conutry.year>StartYear]
    min_year = temp_Conutry['year'].min()
    max_year = temp_Conutry['year'].max()
    years = range(min_year, max_year + 1)
    total_temps=[]
    for year in years:
        curr_years_data = temp_Conutry[temp_Conutry['year'] == year]
        total_temps.append(curr_years_data.AverageTemperature.mean())
      
      
    
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
plt.figure()
plt.plot(years,total_temps,label='Average Temperature per Year',color='black')
#plt.plot(years, moving_av_short_vec, label='Moving Average short period', color='red')
#plt.plot(years, moving_av_long_vec, label='Moving Average long period', color='blue')
plt.ylabel('Average temperature')
plt.xlabel('Year')
string='Average temperature {}'.format(Location)
plt.title(string)

##################################Doing Basic linear Regression
y=np.array(total_temps[0:Datapoints])
x=np.array(years[0:Datapoints])-(StartYear+1)
x_mean=np.mean(x)
y_mean=np.mean(y)

slope=(x-x_mean).dot(y-y_mean)/((x-x_mean).dot(x-x_mean)) #slope of Regression
intercept=y_mean-slope*x_mean #intercept of linear regression
print('Basic linear regression:')
print('Intercept:',intercept,'Slope:', slope)

xplot=(np.array(years)-(StartYear+1))
LinRegr_Est=slope*(np.array(years)-(StartYear+1))+intercept
plt.plot(years,LinRegr_Est,label='Basic Linear Regression', color='green')

#Doing basic polynomial Regression using python's in-built function
z=np.polyfit(x,y,3)
print('Coefficients for basic polynomial Regression')
print(z)
Poly_Regr=z[3]+z[2]*xplot+z[1]*xplot**2+z[0]*xplot**3
intercept_poly=z[3]
plt.plot(years,Poly_Regr,label='Polynomial Regression', color='red')
Res0=sum((LinRegr_Est-total_temps)**2) #Calculating Residual
print('Sum of Squares standard linear regression:', Res0)
Res0=sum((Poly_Regr-total_temps)**2) #Calculating Residual
print('Sum of Squares polynomial regression:', Res0)

#Define Prior
def log_gaussian_prior(mu_0 = 0, sigma_0 = 5):
        ans = lambda mu: -0.5 * np.log(2 * np.pi) - np.log( sigma_0 ) - 0.5 * (mu - mu_0)**2 / sigma_0**2
        return ans

def log_multi_gaussian_prior(mu_0 = np.array([0,0,0,0]), Cov = np.array([[10,0,0,0],[0,1,0,0],[0,0,0.1,0],[0,0,0,0.01]])):
        k=np.sqrt(Cov.size)
        B=np.linalg.inv(Cov)
        ans = lambda mu: -0.5*k * np.log(2 * np.pi) - 0.5*np.linalg.det(Cov) \
         - 0.5 * np.transpose((mu - mu_0))@B@(mu-mu_0)
        return ans

def log_gamma_prior_sigma(a_0 = 1, b_0 = 1):
        ans = lambda sigma: a_0 * np.log(b_0) - np.real(sp.special.loggamma(a_0)) + (a_0 - 1) * np.log( sigma ) - b_0 * sigma
        return ans
    
def log_gamma_prior_d(a_0 = 2, b_0 = 0.5):
        ans = lambda d: a_0 * np.log(b_0) - np.real(sp.special.loggamma(a_0)) + (a_0 - 1) * np.log( d ) - b_0 * d
        return ans  

def log_prior():
    log_prior_mu    = log_multi_gaussian_prior()
    log_prior_sigma = log_gamma_prior_sigma()
    log_prior_d     = log_gamma_prior_d()
    ans = lambda mu, sigma, d: log_prior_mu(mu) + log_prior_sigma(sigma) + log_prior_d(d)
    return ans


def LogLikli(x,y):
    def LogLikli_dens(mu,sigma,d):
        xvec=np.array([1,x,x**2,x**3])
        return  np.real(-np.log(sigma)+sp.special.loggamma((d+1)/2)-0.5*np.log(d*np.pi)-sp.special.loggamma(d/2)\
                -(d+1)/2*np.log(1+(y-mu.dot(xvec))**2/sigma**2/d))
    return LogLikli_dens


LogTest=LogLikli(2000,9)
#print(LogTest(np.array([0.01,0.1,1]),1,4))

Liklilist=[]
count=0

for xi,yi in zip(x,y):
    #print(xi,yi)
    count=count+1
    L1=LogLikli(xi,yi)
    Liklilist.append(L1)
    
Lik=sum_handles3(Liklilist)   
#print(Lik(np.array([1,1,1]),1,4))

#Calculate Posterior

def Log_Post(mu,sigma,d):
    log_prior_A=log_prior()
    return log_prior_A(mu,sigma,d)+Lik(mu,sigma,d) 

def Log_Post_neg(mu,sigma,d):
    log_prior_A=log_prior()
    return -(log_prior_A(mu,sigma,d)+Lik(mu,sigma,d))


# Printing what you like is important for success
for i in range(1):
    print("Eier~Eier~Eier")
    
##########################################################################
if Method=='MH':
    #Metropolis Hastigns
    def MH_Gaus(Log_Post, stepsize=np.array([10,1,0.1,0.01,0.1,0.1,0.1]), No_samples=1000,init=np.array([0.0,0,0,0,1,4])):
        chain=np.zeros([No_samples,6])
        count=0
        chain[0,:]=init
        for i in range(No_samples-1):
            curr=chain[i,:]
            prop=np.zeros([6,])
            prop[0:4]=curr[0:4]+stepsize[0:4]*np.random.multivariate_normal(np.zeros([4,]),np.eye(4))
            prop[[-2,-1]]=np.exp(np.log(curr[[-2,-1]])+stepsize[[-2,-1]]*np.random.standard_normal(2))
            v2=stepsize[-1]**2
            v1=stepsize[-2]**2
            h21=1/(np.sqrt(2*np.pi*v1))/curr[-2]*np.exp(-0.5*(np.log(prop[-2])-np.log(curr[-2]))**2/v1)
            h22=1/(np.sqrt(2*np.pi*v1))/prop[-2]*np.exp(-0.5*(np.log(curr[-2])-np.log(prop[-2]))**2/v1)
            h31=1/(np.sqrt(2*np.pi*v2))/curr[-1]*np.exp(-0.5*(np.log(prop[-1])-np.log(curr[-1]))**2/v2)
            h32=1/(np.sqrt(2*np.pi*v2))/prop[-1]*np.exp(-0.5*(np.log(curr[-1])-np.log(prop[-1]))**2/v2)
            Check=min(1,np.exp(Log_Post(prop[0:4],prop[-2],prop[-1])-Log_Post(curr[0:4],curr[-2],curr[-1]))*h21*h31/h22/h32)
            U=np.random.uniform(0,1,1)
            #print(prop,Log_Post(prop[0:4],prop[-2],prop[-1]),Check,U)
            if U <Check:
                chain[i+1,:]=prop
                count=count+1
            else:
                chain[i+1,:]=curr
        Ratio=count/No_samples
        return chain,Ratio
    
    start = time.time()
    base=0.01
    a=np.array([0,1,2,3],float)
    steps=base**a
    steps=np.array([0.001,1e-4,1e-6,1e-8])
    steps1=np.append(steps,[0.01,1])
    Start_Stats=2000 #Value to define where to consider values of the Markov chain for final statistics
    runs=20000
    chain,Ratio=MH_Gaus(Log_Post,stepsize=steps1,\
                        No_samples=runs,init=np.array([intercept_poly,0,0,0,0.2,4]))
    end = time.time()
    
    print('Acceptance Ratio:', Ratio)
    a0=np.mean(chain[Start_Stats:,0])
    a1=np.mean(chain[Start_Stats:,1])
    a2=np.mean(chain[Start_Stats:,2])
    a3=np.mean(chain[Start_Stats:,3])
    a4=np.mean(chain[Start_Stats:,4])
    a5=np.mean(chain[Start_Stats:,5])

    xval=np.array(years)-StartYear
    PolRegr_Est=a3*xval**3+a2*xval**2+a1*xval+a0
    plt.plot(years,PolRegr_Est,label=' Bayesian Polynomial Regression', color='orange')
    #PolRegr_Estb=a3b*(np.array(years)-1831)**3+a2b*(np.array(years)-1831)**2+a1b*(np.array(years)-1831)+a0b
    #plt.plot(years,PolRegr_Estb,label='Polynomial Regression', color='yellow')
    plt.legend(loc='best',frameon=1)
    plt.savefig('Graph.png')
    
    print('Coefficients for polynomial regression:')
    print(a0,a1,a2,a3)
    Res=sum((PolRegr_Est-total_temps)**2) #Calculating new residuals
    print('Variance Sigma:',a4)
    print('Degrees of Freedom',a5)
    print('Sum of Squares Model:',Res)
    
    Time=end-start
    print('Runtime:',Time,'Sekunden')
    plt.figure()
    plt.hist(chain[Start_Stats:,0],density=True,label='mu1',bins=20)
    plt.xlabel('$a_0$')
    plt.savefig('Hist_a0.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,1],density=True,bins=20)
    plt.xlabel('$a_1$')
    plt.savefig('Hist_a1.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,2],density=True,bins=20)
    plt.xlabel('$a_2$')
    plt.savefig('Hist_a2.png')
    #plt.figure()
    fig,ax=plt.subplots(1,1)
    ax.hist(chain[Start_Stats:,3],density=True,bins=20)
    plt.xlabel('$a_3$')
    fig.autofmt_xdate()
    plt.show()
    plt.savefig('Hist_a3.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,4],density=True,bins=20)
    plt.xlabel('$\sigma$')
    plt.savefig('Hist_sigma.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,5],density=True,bins=20)
    plt.xlabel('Degrees of Freedom')
    plt.savefig('Hist_d.png')
    
    plt.figure()
    plt.plot(chain[:,0],'*')
    plt.xlabel('Samples')
    plt.ylabel('$a_0$')
    plt.savefig('MH_a0.png')
    plt.figure()
    plt.plot(chain[:,1],'*')
    plt.xlabel('Samples')
    plt.ylabel('$a_1$')
    plt.savefig('MH_a1.png')
    plt.figure()
    plt.plot(chain[:,2],'*')
    plt.xlabel('Samples')
    plt.ylabel('$a_2$')
    plt.savefig('MH_a2.png')
    plt.figure()
    plt.plot(chain[:,3],'*')
    plt.xlabel('Samples')
    plt.ylabel('$a_3$')
    plt.savefig('MH_a3.png')
    plt.figure()
    plt.plot(chain[:,4],'*')
    plt.xlabel('Samples')
    plt.ylabel('$\sigma$')
    plt.savefig('MH_sigma.png')
    plt.figure()
    plt.plot(chain[:,5],'*')
    plt.xlabel('d')
    plt.savefig('MH_d.png')
    
    
elif Method=='GVA': 
    #Gaussian Variatonal Approximation
    #Numerical Approxiimation of the Gradient
    def NumGrad6(f,P,sig,d,h=0.01):
        P1=np.array([0,0,0,0],float)
        P1[0]=P[0]+h
        P1[1:]=P[1:]
        G1=(f(P1,sig,d)-f(P,sig,d))/h
        P1=np.array([0,0,0,0],float)
        P1[1]=P[1]+h
        P1[[0,2,3]]=P[[0,2,3]]
        G2=(f(P1,sig,d)-f(P,sig,d))/h
        P1=np.array([0,0,0,0],float)
        P1[2]=P[2]+h
        P1[[0,1,3]]=P[[0,1,3]]
        G3=(f(P1,sig,d)-f(P,sig,d))/h
        P1=np.array([0,0,0,0],float)
        P1[3]=P[3]+h
        P1[[0,1,2]]=P[[0,1,2]]
        G4=(f(P1,sig,d)-f(P,sig,d))/h
        sig1=sig+h
        G5=(f(P,sig1,d)-f(P,sig,d))/h
        d1=d+h
        G6=(f(P,sig,d1)-f(P,sig,d))/h
        Grad=np.array([G1,G2,G3,G4,G5,G6])
        return Grad

    #Numerical second derivative for GVA where covariance is diagonal
    def NumLfact(f,P,sig,d,h=0.001):
        L1=(f(np.array([P[0]+2*h,P[1],P[2],P[3]]),sig,d)-2*f(np.array([P[0]+h,P[1],P[2],P[3]]),sig,d)\
            +f(np.array([P[0],P[1],P[2],P[3]]),sig,d))/h**2
        L2=(f(np.array([P[0],P[1]+2*h,P[2],P[3]]),sig,d)-2*f(np.array([P[0],P[1]+h,P[2],P[3]]),sig,d)\
            +f(np.array([P[0],P[1],P[2],P[3]]),sig,d))/h**2
        L3=(f(np.array([P[0],P[1],P[2]+2*h,P[3]]),sig,d)-2*f(np.array([P[0],P[1],P[2]+h,P[3]]),sig,d)\
            +f(np.array([P[0],P[1],P[2],P[3]]),sig,d))/h**2
        L4=(f(np.array([P[0],P[1],P[3],P[3]+2*h]),sig,d)-2*f(np.array([P[0],P[1],P[3],P[3]+h]),sig,d)\
            +f(np.array([P[0],P[1],P[3],P[3]]),sig,d))/h**2
        L5=(f(np.array([P[0],P[1],P[3],P[3]]),sig+2*h,d)-2*f(np.array([P[0],P[1],P[3],P[3]]),sig+h,d)\
            +f(np.array([P[0],P[1],P[3],P[3]]),sig,d))/h**2
        L6=(f(np.array([P[0],P[1],P[3],P[3]]),sig,d+2*h)-2*f(np.array([P[0],P[1],P[3],P[3]]),sig,d+h)\
            +f(np.array([P[0],P[1],P[3],P[3]]),sig,d))/h**2
        return np.array([L1,L2,L3,L4,L5,L6])

    fct=Log_Post_neg
    def GVA(mu0=np.array([0,0,0,0,1,1]),L0=np.array([-1,-2,-3,-4,-3,-3],float),\
            NoSamples=1000,runs=100,scale=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5]),\
            scale2=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5])):
        l=NoSamples
        muG=mu0
        L=L0
        dim=np.sqrt(L.size)
        for i in range(runs):
            eta=np.random.multivariate_normal(mean=np.array([0,0,0,0,0,0],float),cov=np.eye(6),size=l)
            Postvec=np.zeros([l,1],float)
            Grad_Post=np.zeros([l,6],float)
            L_Post=np.zeros([l,6],float)
            count=0
            #Computing stochastic approximation for gradient
            for j in range(l):
                Lexp=np.exp(L)
                Input=muG+Lexp@eta[j]
                #print('mu',muG,'L',Lexp,'eta',eta[j],'I',Input)
                coeffs=Input[0:4]
                sigma=Input[-2]
                d=Input[-1]
                #print(coeffs,sigma,d)
                if sigma>0 and d>0: #Make sure that sigma and d are positive
                    Postvec[j]=fct(coeffs,sigma,d)
                    Grad_Post[j,]=NumGrad6(fct,coeffs,sigma,d,h=0.0001)
                    L_Post[j,]=NumLfact(fct,coeffs,sigma,d,h=0.00001)
                    #print(Grad_Post)
                    count=count+1
            if count>0:
                ELBO=-1/count*sum(Postvec)+dim/2*np.log(2*np.pi*np.e)+np.sum(L)
                Grad_ELBO=(-1/count)*np.sum(Grad_Post,axis=0)#Update for mu
                AvgL=(1/count)*np.sum(L_Post,axis=0)
                #print('Avg',AvgL)
                L_Elbo0=-np.exp(2*L[0])*AvgL[0]+1        #Computing values for L matrix update
                L_Elbo1=-np.exp(2*L[1])*AvgL[1]+1
                L_Elbo2=-np.exp(2*L[2])*AvgL[2]+1        #Computing values for L matrix update
                L_Elbo3=-np.exp(2*L[3])*AvgL[3]+1
                L_Elbo4=-np.exp(2*L[4])*AvgL[4]+1        #Computing values for L matrix update
                L_Elbo5=-np.exp(2*L[5])*AvgL[5]+1
                L_Elbo=np.array([L_Elbo0,L_Elbo1,L_Elbo2,L_Elbo3,L_Elbo4,L_Elbo5])
                muG=muG+((1/(i+1))*Grad_ELBO*scale) #SGD step
                L=L+(1/(i+1)*L_Elbo*scale2) # SGD step
            else:
                muG=muG
                L=L
            #print('mu',muG,ELBO,'L',L)
        return muG,L
    start = time.time()
    mu,L=GVA(mu0=np.array([intercept_poly,0,0,0,1,4]),L0=np.array([-1,-2,-3,-4,-3,-3],float),\
        NoSamples=100,runs=10,scale=np.array([1e-1,1e-5,1e-7,1e-9,1e-4,1e-4]),\
        scale2=np.array([1e-2,1e-5,1e-7,1e-11,1e-2,1e-1]))
    print('Final mu:',mu)
    Cov1=sp.linalg.expm(np.diag(L))#Covariacne for GVA with diagonal matrix
    Cov=Cov1@np.transpose(Cov1)
    print('Diagonal of Covariance Matrix:',np.diag(Cov))
    
    end = time.time()
    Time=end-start
    print('Runtime:', Time)
    xval=np.array(years)-StartYear
    Bays_poly=mu[0]+mu[1]*xval+mu[2]*xval**2+mu[3]*xval**3
    plt.plot(years,Bays_poly, label='Bayesian Polynomial Regression', color='orange')
    plt.legend(loc='best',frameon=1)
    
    print('Coefficients for polynomial regression:')
    print(mu[0],mu[1],mu[2],mu[3])
    Res=sum((Bays_poly-total_temps)**2) #Calculating new residuals
    print('Variance Sigma:',mu[4])
    print('Degrees of Freedom',mu[5])
    print('Sum of Squares Model:',Res)
        
    #Defining Standard Gaussian
    def uni_Gaus(mean,sigma):
        def uni_Gaus_unnorm_dens(x):
            return 1/(np.sqrt(2*sigma**2*np.pi))*np.exp(-(x-mean)**2/(2*sigma**2))
        return uni_Gaus_unnorm_dens
    
    #Plotting distribution of sigma and d
    plotN=100
    Distr_sigma=uni_Gaus(mu[-2],Cov[(-2,-2)])
    sigmavec=np.zeros([plotN,])
    sigmaplot=np.zeros([plotN,])
    Distr_d=uni_Gaus(mu[-1],Cov[(-1,-1)])
    dvec=np.zeros([plotN,])
    dplot=np.zeros([plotN,])
    for i in range(plotN):
        d=mu[-1]-4*Cov[(-1,-1)]+i/plotN*8*Cov[(-1,-1)]
        dvec[i]=d
        dplot[i]=Distr_d(d)
        sigma=mu[-2]-4*Cov[(-2,-2)]+i/plotN*8*Cov[(-2,-2)]
        sigmavec[i]=sigma
        sigmaplot[i]=Distr_sigma(sigma)
      
    plt.figure()
    plt.plot(dvec,dplot)
    plt.xlabel('Degrees of Freedom')
    plt.title('Approximation of Marginal Distribution')
    plt.figure()
    plt.plot(sigmavec,sigmaplot)
    plt.xlabel('Sigma')
    plt.title('Approximation of Marginal Distribution')
   
elif Method=='GVA2':  
    #GVA with full Hessian
    def NumHes6(f,P,h=0.01):
        Hes=np.zeros([6,6])
        for i1 in range(6):
                for i2 in range(6):
                    P1=np.array([P[0],P[1],P[2],P[3],P[4],P[5]])
                    P2a=np.array([P[0],P[1],P[2],P[3],P[4],P[5]])
                    P2b=np.array([P[0],P[1],P[2],P[3],P[4],P[5]])
                    P1[i1]=P1[i1]+h
                    P2a[i1]=P2a[i1]+h
                    P1[i2]=P1[i2]+h
                    P2b[i1]=P2b[i1]+h
                    #print(P1,P2a,P2b,P)
                    H1=f(np.array([P1[0],P1[1],P[2],P[3]]), P1[-2], P1[-1])
                    H2a=f(np.array([P2a[0],P2a[1],P2a[2],P2a[3]]), P2a[-2], P1[-1])
                    H2b=f(np.array([P2b[0],P2b[1],P2b[2],P2b[3]]), P2b[-2], P2b[-1])
                    H3=f(np.array([P[0],P[1],P[2],P[3]]), P[-2], P[-1])
                    H=(H1-H2a-H2b+H3)/(h**2)
                    Hes[i1,i2]=H
                    Hes[i2,i1]=H
        return Hes

    #Numerical Approxiimation of the Gradient
    def NumGrad6(f,P,sig,d,h=0.01):
        P1=np.array([0,0,0,0],float)
        P1[0]=P[0]+h
        P1[1:]=P[1:]
        G1=(f(P1,sig,d)-f(P,sig,d))/h
        P1=np.array([0,0,0,0],float)
        P1[1]=P[1]+h
        P1[[0,2,3]]=P[[0,2,3]]
        G2=(f(P1,sig,d)-f(P,sig,d))/h
        P1=np.array([0,0,0,0],float)
        P1[2]=P[2]+h
        P1[[0,1,3]]=P[[0,1,3]]
        G3=(f(P1,sig,d)-f(P,sig,d))/h
        P1=np.array([0,0,0,0],float)
        P1[3]=P[3]+h
        P1[[0,1,2]]=P[[0,1,2]]
        G4=(f(P1,sig,d)-f(P,sig,d))/h
        sig1=sig+h
        G5=(f(P,sig1,d)-f(P,sig,d))/h
        d1=d+h
        G6=(f(P,sig,d1)-f(P,sig,d))/h
        Grad=np.array([G1,G2,G3,G4,G5,G6])
        return Grad

    fct=Log_Post_neg
    def GVA2(mu0=np.array([0,0,0,0,1,1]),L0=np.array([-1,-2,-3,-4,-3,-3],float),\
            NoSamples=1000,runs=100,scale=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5]),\
            scale2=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5])):
        l=NoSamples
        muG=mu0
        L=L0
        dim=np.sqrt(L.size)
        for i in range(runs):
            eta=np.random.multivariate_normal(mean=np.array([0,0,0,0,0,0],float),cov=np.eye(6),size=l)
            Postvec=np.zeros([l,1],float)
            Grad_Post=np.zeros([l,6],float)
            L_Post=np.zeros([l,6,6],float)
            count=0
            #Computing stochastic approximation for gradient
            for j in range(l):
            #print(muG,L,eta[j])
                Input=muG+sp.linalg.expm(L)@eta[j]
                coeffs=Input[0:4]
                sigma=Input[-2]
                d=Input[-1]
                #print(coeffs,sigma,d)
                if sigma>0 and d>0: #Make sure that sigma and d are positive
                    Postvec[j]=fct(coeffs,sigma,d)
                    Grad_Post[j,]=NumGrad6(fct,coeffs,sigma,d,h=0.0001)
                    L_Post[j,]=NumHes6(fct,Input,h=0.001)
                    #print(Grad_Post)
                    count=count+1
            if count>0:
                ELBO=-1/count*sum(Postvec)+dim/2*np.log(2*np.pi*np.e)+np.sum(L)
                Grad_ELBO=(-1/count)*np.sum(Grad_Post,axis=0)
                AvgL=(1/count)*np.sum(L_Post,axis=0)
                #print('Avg',AvgL)
                L_Elbo1=sp.linalg.expm(2*L)@AvgL
                L_Elbo=-0.5*(L_Elbo1+np.transpose(L_Elbo1))+np.eye(6) #symmetrise L matrix
                #print(L,'Elbo',L_Elbo)
                #-0.5*(sp.linalg.expm(2*L)@((1/l)*sum(L_Post))+np.transpose(sp.linalg.expm(2*L)@((1/l)*sum(L_Post))))+np.eye(2)
                #print(Grad_ELBO,muG)
                muG=muG+((1/(i+1))*Grad_ELBO*scale) #SGD step
                L=L+(1/(i+1)*L_Elbo*scale2) # SGD step
            else:
                muG=muG
                L=L
            #print(muG,ELBO,L,count)
        return muG,L

    start=time.time()
    mu,L=GVA2(mu0=np.array([intercept_poly,0,0,0,1,4]),L0=np.diag([-1,-2,-3,-4,-3,-3]),\
        NoSamples=20,runs=200,scale=np.array([1e-2,1e-5,1e-7,1e-13,1e-4,1e-4]),\
            scale2=1e-8*np.ones([6,6])+1e-8*np.diag(np.ones(6))+np.diag([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5]))
    print('Final mu:',mu)
    Cov1=sp.linalg.expm(L)#Covariacne for GVA with diagonal matrix
    Cov=Cov1@np.transpose(Cov1)
    print('Covariance Matrix:',Cov)
    
    end = time.time()
    Time=end-start
    print('Runtime:', Time)
    xval=np.array(years)-StartYear
    Bays_poly=mu[0]+mu[1]*xval+mu[2]*xval**2+mu[3]*xval**3
    plt.plot(years,Bays_poly, label='Bayesian Polynomial Regression', color='orange')
    plt.legend(loc='best',frameon=1)
    
    print('Coefficients for polynomial regression:')
    print(mu[0],mu[1],mu[2],mu[3])
    Res=sum((Bays_poly-total_temps)**2) #Calculating new residuals
    print('Variance Sigma:',mu[4])
    print('Degrees of Freedom',mu[5])
    print('Sum of Squares Model:',Res)
    
    #Defining Standard Gaussian
    def uni_Gaus(mean,sigma):
        def uni_Gaus_unnorm_dens(x):
            return 1/(np.sqrt(2*sigma**2*np.pi))*np.exp(-(x-mean)**2/(2*sigma**2))
        return uni_Gaus_unnorm_dens
    
  #Plotting distribution of sigma and d
    plotN=100
    Distr_sigma=uni_Gaus(mu[-2],Cov[(-2,-2)])
    sigmavec=np.zeros([plotN,])
    sigmaplot=np.zeros([plotN,])
    Distr_d=uni_Gaus(mu[-1],Cov[(-1,-1)])
    dvec=np.zeros([plotN,])
    dplot=np.zeros([plotN,])
    for i in range(plotN):
        d=mu[-1]-4*Cov[(-1,-1)]+i/plotN*8*Cov[(-1,-1)]
        dvec[i]=d
        dplot[i]=Distr_d(d)
        sigma=mu[-2]-4*Cov[(-2,-2)]+i/plotN*8*Cov[(-2,-2)]
        sigmavec[i]=sigma
        sigmaplot[i]=Distr_sigma(sigma)
      
    plt.figure()
    plt.plot(dvec,dplot)
    plt.xlabel('Degrees of Freedom')
    plt.title('Approximation of Marginal Distribution')
    plt.figure()
    plt.plot(sigmavec,sigmaplot)
    plt.xlabel('Sigma')
    plt.title('Approximation of Marginal Distribution')

#############################################################################    
elif Method=='Laplace':
    #Laplace Approximation
    def NumHes6(f,P,h=0.01):
        Hes=np.zeros([6,6])
        for i1 in range(6):
                for i2 in range(6):
                    P1=np.array([P[0],P[1],P[2],P[3],P[4],P[5]])
                    P2a=np.array([P[0],P[1],P[2],P[3],P[4],P[5]])
                    P2b=np.array([P[0],P[1],P[2],P[3],P[4],P[5]])
                    P1[i1]=P1[i1]+h
                    P2a[i1]=P2a[i1]+h
                    P1[i2]=P1[i2]+h
                    P2b[i1]=P2b[i1]+h
                    #print(P1,P2a,P2b,P)
                    H1=f(np.array([P1[0],P1[1],P[2],P[3]]), P1[-2], P1[-1])
                    H2a=f(np.array([P2a[0],P2a[1],P2a[2],P2a[3]]), P2a[-2], P1[-1])
                    H2b=f(np.array([P2b[0],P2b[1],P2b[2],P2b[3]]), P2b[-2], P2b[-1])
                    H3=f(np.array([P[0],P[1],P[2],P[3]]), P[-2], P[-1])
                    H=(H1-H2a-H2b+H3)/(h**2)
                    Hes[i1,i2]=H
                    Hes[i2,i1]=H
        return Hes
    start=time.time()
    Nopoints2=40000
    Points=np.zeros([Nopoints2,6])
    Base=np.array([intercept_poly-1,-0.01,-0.001,-0.0001,0,0])
    Factor=np.array([2,0.02,0.002,0.0002,3,20])
    Postvec=np.zeros(Nopoints2)
    for k in range(Nopoints2):
        R=np.random.random(6)
        Point=Base+R*Factor
        Points[k]=Point
        Postvec[k]=Log_Post(np.array([Point[0],Point[1],Point[2],Point[3]]), Point[-2], Point[-1])
        

    Mindx=np.argmax(Postvec)
    MaxPoint=Points[Mindx]
    print('Mean:',MaxPoint)
    beta=-NumHes6(Log_Post,MaxPoint)
    Cov=np.linalg.inv(beta)
    print('Covariance:', Cov)
    end=time.time()
    xval=np.array(years)-StartYear
    Bays_poly=MaxPoint[0]+MaxPoint[1]*xval+MaxPoint[2]*xval**2+MaxPoint[3]*xval**3
    plt.plot(years,Bays_poly, label='Bayesian Polynomial Regression', color='orange')

    plt.legend(loc='best',frameon=1)
    plt.savefig('Graph.png')
    print('Coefficients for polynomial regression:')
    print(MaxPoint[0],MaxPoint[1],MaxPoint[2],MaxPoint[3])
    Res=sum((Bays_poly-total_temps)**2) #Calculating new residuals
    print('Variance Sigma:',MaxPoint[4])
    print('Degrees of Freedom',MaxPoint[5])
    print('Sum of Squares Model:',Res)
    
    muL=np.array([MaxPoint[1],MaxPoint[0]])
    #sigmaL=NumHes3a(LogPost,0.01,MaxPoint[2],muL)
    #sigmaL2=NumHes3a(Post,0.01,MaxPoint[2],muL)

    Time=end-start
    print('Runtime:',Time)
    
    
  #  n=100
  #  for i1 in range(n):
  #      a2=-0.001*i1/n*0.002
  #      for i2 in range(n):
  #          a3=-0.0001*i2/n*0.0002   
   #         Postvec2[k]=Log_Post(np.array([Point[0],Point[1],Point[2],Point[3]]), Point[-2], Point[-1])
            
    Nopoints=10
    avec=np.zeros(Nopoints**2)
    bvec=np.zeros(Nopoints**2)
    Postvec=np.zeros(Nopoints**2)
    n=Nopoints
    for k1 in range(Nopoints):
        a=1e-07+k1/Nopoints*4e-07
        for k2 in range(Nopoints):
            indx=Nopoints*k1+k2
            b=5.5+k2/Nopoints*4
            avec[indx]=a
            bvec[indx]=b
            Postvec[indx]=np.exp(Log_Post(np.array([b,1.39e-02,-1.12e-04,a]), MaxPoint[-2], MaxPoint[-1]))
            
    xplot = np.reshape(avec, (Nopoints, Nopoints))
    yplot = np.reshape(bvec, (Nopoints, Nopoints))
    zplot = np.reshape(Postvec, (Nopoints, Nopoints))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    Postplot=ax.plot_surface(xplot, yplot, zplot, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_xlabel('gradient')
    ax.set_ylabel('constant')
    ax.set_zlabel('Unscaled Posterior')
    fig.colorbar(Postplot, shrink=0.5, aspect=5)
    
    plt.show()
        
#    Nopoints=int(np.sqrt(Nopoints2))
 ##   xplot = np.reshape(Points[:,1], (Nopoints, Nopoints))
  #  yplot = np.reshape(Points[:,2], (Nopoints, Nopoints))
  #  zplot = np.reshape(Postvec, (Nopoints, Nopoints))
   ## fig = plt.figure()
  #  ax = fig.gca(projection='3d')
    
  ##  Postplot=ax.plot_surface(xplot, yplot, zplot, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
  #  ax.set_xlabel('Tgradient')
  #  ax.set_ylabel('constant')
  #  ax.set_zlabel('Unscaled Posterior')
  #  fig.colorbar(Postplot, shrink=0.5, aspect=5)
    
  #  plt.show()
    