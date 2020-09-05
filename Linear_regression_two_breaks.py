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

plt.close('all')
import warnings
warnings.filterwarnings('ignore')

#####################################################################
#Description: 
#This Program fits a basyesian linear regression with two break points
#to the temeprature on different locations in the world. The location (countries) can be chosenn 
#with the Location parameter. This means that there are two parameters for the break years and
# 4 (intercept+slope+slope+slope) parameters for the linear regressions in between. It is best understood
#by looking at the resulting graphs. Those parameters are assumed to be normally distributed. In addition we work
# with a student noise model which gives an overall of 8 parameters to be estimated.
#
#Three different approximation methods
#have been implemented wich can be chosen through the Method parameter. 
#Parameteres for different models are set in appropriate section. Different location may require 
#a change in the stepsize for MH or the scale for GVA.

#As the Output the script first gives the parameters of basic linear regression and the sum of least squares.
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
            #A huge difference can be seen here when GVA2 is used because this has quite a long runtime.
            # Due to round-off errors, wometimes matrices appaer to be singular.
            
   


########################Function needed in code
def sum_handles6(handles_list):
    def aux(x1,x2,x3,x4,x5,x6):
        temp=0
        for f in handles_list:
            temp=temp+f(x1,x2,x3,x4,x5,x6)
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

##################################Doing Basic linear Regression as a comparative result
y=np.array(total_temps[0:Datapoints])
x=np.array(years[0:Datapoints])-(StartYear+1)
x_mean=np.mean(x)
y_mean=np.mean(y)

#Using known formuales to calculate parameters
slope=(x-x_mean).dot(y-y_mean)/((x-x_mean).dot(x-x_mean)) #slope of Regression
intercept=y_mean-slope*x_mean #intercept of linear regression
print('Basic linear regression:')
print('Intercept:',intercept,'Slope:', slope)

xplot=(np.array(years)-(StartYear+1))
LinRegr_Est=slope*(np.array(years)-(StartYear+1))+intercept
plt.plot(years,LinRegr_Est,label='Basic Linear Regression', color='green')
Res0=sum((LinRegr_Est-total_temps)**2) #Calculating Residual
print('Sum of Squares standard linear regression:', Res0)

##############################Define Prior by defining priors for different parameters and taking the sum of their logs
def log_gaussian_prior(mu_0 = 0, sigma_0 = 5):
        ans = lambda mu: -0.5 * np.log(2 * np.pi) - np.log( sigma_0 ) - 0.5 * (mu - mu_0)**2 / sigma_0**2
        return ans

def log_multi_gaussian_prior(mu_0 = np.array([-10,0]), Cov = np.array([[10,0],[0,5]])):
        k=np.sqrt(Cov.size)
        B=np.linalg.inv(Cov)
        ans = lambda mu: -0.5*k * np.log(2 * np.pi) - 0.5*np.linalg.det(Cov) \
         - 0.5 * np.transpose((mu - mu_0))@B@(mu-mu_0)
        return ans

def log_unif_prior(begin=1,end=2016-StartYear-1,n=2):
    ans=lambda breaks: n*np.log((1/(end-begin)))
    return ans

def log_gamma_prior_sigma(a_0 = 1, b_0 = 1):
        ans = lambda sigma: a_0 * np.log(b_0) - np.real(sp.special.loggamma(a_0)) + (a_0 - 1) * np.log( sigma ) - b_0 * sigma
        return ans
    
def log_gamma_prior_d(a_0 = 2, b_0 = 0.5):
        ans = lambda d: a_0 * np.log(b_0) - np.real(sp.special.loggamma(a_0)) + (a_0 - 1) * np.log( d ) - b_0 * d
        return ans  

def log_prior():
    log_prior_breaks = log_unif_prior() #prior for breakpoint
    log_prior_mu1    = log_multi_gaussian_prior() # prior for intercept and slope for first interval
    log_prior_mu2 =log_multi_gaussian_prior() # slope for second interval
    log_prior_mu3 =log_multi_gaussian_prior()# slope for third interval
    log_prior_sigma = log_gamma_prior_sigma()#sigma as variance of noise
    log_prior_d     = log_gamma_prior_d() #degree of freedom for student noise
    def ans(breaks, mu1, mu2a, mu3a, sigma, d):
        mu2=np.array([mu1[0]+breaks[0]*mu1[1],mu2a])
        mu3=np.array([mu1[0]+breaks[0]*mu1[1]+mu2a[0]*(breaks[1]-breaks[0]),mu3a])
        return log_prior_breaks(breaks)+log_prior_mu1(mu1)+\
         log_prior_mu2(mu2)+log_prior_mu3(mu3)\
         +log_prior_sigma(sigma) + log_prior_d(d)
    return ans

P=log_prior()
#print('P',P(np.array([50,82]),np.array([7,0.01]),np.array([6]),np.array([6]),1,4))

################Liklihood calculation
#Compute Liklihood as a linear regression model with student noise
def LogLikli(x,y):
    def LogLikli_dens(breaks,mu1,mu2a,mu3a,sigma,d):
        if x <=breaks[0]:
            xvec=np.array([1,x])
            return  np.real(-np.log(sigma)+sp.special.loggamma((d+1)/2)-0.5*np.log(d*np.pi)-sp.special.loggamma(d/2)\
                    -(d+1)/2*np.log(1+(y-mu1.dot(xvec))**2/sigma**2/d))
        elif x<=breaks[1]:
            xvec=np.array([1,x-breaks[0]])
            mu2=np.array([mu1[0]+mu1[1]*breaks[0],mu2a])
            return  np.real(-np.log(sigma)+sp.special.loggamma((d+1)/2)-0.5*np.log(d*np.pi)-sp.special.loggamma(d/2)\
                    -(d+1)/2*np.log(1+(y-mu2.dot(xvec))**2/sigma**2/d))
        else:
            xvec=np.array([1,x-breaks[1]])
            mu3=np.array([mu1[0]+breaks[0]*mu1[1]+mu2a[0]*(breaks[1]-breaks[0]),mu3a])
            return  np.real(-np.log(sigma)+sp.special.loggamma((d+1)/2)-0.5*np.log(d*np.pi)-sp.special.loggamma(d/2)\
                    -(d+1)/2*np.log(1+(y-mu3.dot(xvec))**2/sigma**2/d))
    return LogLikli_dens


LogTest=LogLikli(x[100],y[100])
#print('L',LogTest(np.array([50,100]),np.array([10,0.01]),np.array([1]),np.array([1]),1,4))

Liklilist=[]
count=0

for xi,yi in zip(x,y):
    #print(xi,yi)
    count=count+1
    L1=LogLikli(xi,yi)
    Liklilist.append(L1)


Lik=sum_handles6(Liklilist)   


################################Calculate Posterior

def Log_Post(breaks,mu1,mu2a,mu3a,sigma,d):
    log_prior_A=log_prior()
    return log_prior_A(breaks,mu1,mu2a,mu3a,sigma,d)+Lik(breaks,mu1,mu2a,mu3a,sigma,d) 
#Negative Log-Posterior required for GVA
def Log_Post_neg(breaks,mu1,mu2a,mu3a,sigma,d):
    log_prior_A=log_prior()
    return -(log_prior_A(breaks,mu1,mu2a,mu3a,sigma,d)+Lik(breaks,mu1,mu2a,mu3a,sigma,d))

#print('LogP',Log_Post(np.array([80,150]),np.array([7.8,0.006]),np.array([0.006]),np.array([0.03]),1,4))
# Printing what you like is important for success
for i in range(1):
    print("Eier~Eier~Eier")


##########################Approximation Methods
if Method=='MH':

    #Metropolis Hastigns
    #This is a basic Metropolis Hastings algortihm. The code is based on what was discussed in the lecture and 
    #done in the corresponding lab.
    #Parameters for MH:
    #runs:Number of samples
    #stepsize: as a vector so that it can be different for each variable
    #init: Startingpoint
    def MH_Gaus(Log_Post, stepsize=np.array([10,10,1,0.1,0.1,0.1,0.1,0.1]),\
                No_samples=1000,init=np.array([60,120,8,0,0,0,1,4])):
        chain=np.zeros([No_samples,8])
        count=0
        chain[0,:]=init
        for i in range(No_samples-1):
            curr=chain[i,:]
            prop=np.zeros([8,])
            prop[0:6]=curr[0:6]+stepsize[0:6]*np.random.multivariate_normal(np.zeros([6,]),np.eye(6))
            #print(i,prop[0:2])
            if prop[1]<prop[0] or prop[0]<1 or prop[1]>2013-StartYear: #Makesure that breaks are within range
                chain[i+1,:]=curr           
            else:
                prop[[-2,-1]]=np.exp(np.log(curr[[-2,-1]])+stepsize[[-2,-1]]*np.random.standard_normal(2))
                v2=stepsize[-1]**2
                v1=stepsize[-2]**2
                #Calculating transition probabilities for non-symmetric proposals
                h21=1/(np.sqrt(2*np.pi*v1))/curr[-2]*np.exp(-0.5*(np.log(prop[-2])-np.log(curr[-2]))**2/v1)
                h22=1/(np.sqrt(2*np.pi*v1))/prop[-2]*np.exp(-0.5*(np.log(curr[-2])-np.log(prop[-2]))**2/v1)
                h31=1/(np.sqrt(2*np.pi*v2))/curr[-1]*np.exp(-0.5*(np.log(prop[-1])-np.log(curr[-1]))**2/v2)
                h32=1/(np.sqrt(2*np.pi*v2))/prop[-1]*np.exp(-0.5*(np.log(curr[-1])-np.log(prop[-1]))**2/v2)
                #print(prop[0])
                Check=min(1,np.exp(Log_Post(prop[0:2],prop[2:4],np.array([prop[4]]),np.array([prop[5]]),prop[-2],prop[-1])-\
                                   Log_Post(curr[0:2],curr[2:4],np.array([curr[4]]),np.array([curr[5]]),curr[-2],curr[-1]))*h21*h31/h22/h32)
                U=np.random.uniform(0,1,1) #Bernoulli rv
                #print(prop,Log_Post(np.array([prop[0]]),prop[1:3],prop[3:5],prop[-2],prop[-1]),Check,U)
                if U <Check:
                    chain[i+1,:]=prop
                    count=count+1
                else:
                    chain[i+1,:]=curr
        Ratio=count/No_samples
        return chain,Ratio
    start = time.time()
    
    Start_Stats=2000 #Value to define where to consider values of the Markov chain for final statistics, aka Burn-in period
    runs=5000

    chain,Ratio=MH_Gaus(Log_Post,stepsize=np.array([2,2,0.01,0.0001,0.0001,0.0001,0.01,1]),\
                        No_samples=runs,init=np.array([int(Datapoints/3),int(2*Datapoints/3),intercept,0.01,0.00,0.02,1,4]))
    
    end = time.time()
    Time=end-start
    print('Runtime:',Time,'Sekunden')
    print('Acceptance Ratio Markov Chain:',Ratio)
    
    #Getting values for plotting results and calculating means
    a0=np.mean(chain[Start_Stats:,2])
    a1=np.mean(chain[Start_Stats:,3])
    a2=np.mean(chain[Start_Stats:,4])
    a3=np.mean(chain[Start_Stats:,5])
    a6=np.mean(chain[Start_Stats:,6])
    a7=np.mean(chain[Start_Stats:,7])
    breaks1=int(np.mean(chain[Start_Stats:,0]))
    breaks2=int(np.mean(chain[Start_Stats:,1]))
    
    

    years=np.array(years)
    x1=years[0:breaks1+1]-StartYear
    x2=years[breaks1+1:breaks2+1]-StartYear-breaks1-1
    x3=years[breaks2+1:]-StartYear-breaks2-1
    
    #Calculating approximation based on results from Bayesian regression
    New_reg=np.append(np.append([a0+a1*x1],[a0+a1*(breaks1)+a2*x2]),[a0+a1*(breaks1)+a2*(breaks2-breaks1)+a3*x3])
    plt.plot(years,New_reg, color='red', label='Regression with Breaks')
    plt.scatter([breaks1+StartYear,breaks2+StartYear], [intercept,intercept], s=50, color='orange',label='Breakpoints')
    plt.legend(loc='best',frameon=1)
    plt.savefig('Graph.png')
    
    Res=sum((New_reg-total_temps)**2) #Calculating new residuals
    print('Breakpoints:',breaks1+StartYear,breaks2+StartYear)
    print('Intercept:',a0)
    print('Slopes for different intervals:', a1,a2,a3)
    print('Variance Sigma:',a6)
    print('Degrees of Freedom',a7)
    print('Sum of Squares Model:',Res)
    
    #Plotting Histograms of single parameters
    plt.figure()
    plt.hist(chain[Start_Stats:,0]+StartYear,density=True,label='mu1',bins=20)
    plt.xlabel('First Break')
    plt.savefig('Hist_First_Break.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,1]+StartYear,density=True,bins=20)
    plt.xlabel('Second Break')
    plt.savefig('Hist_second_Break.png')
    plt.figure()
    h2=plt.hist(chain[Start_Stats:,2],density=True,bins=20)
    plt.xlabel('$b$')
    plt.savefig('Hist_b.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,3],density=True,bins=20)
    plt.xlabel('$a_0$')
    plt.savefig('Hist_a0.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,4],density=True,bins=20)
    plt.xlabel('$a_1$')
    plt.savefig('Hist_a1.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,5],density=True,bins=20)
    plt.xlabel('$a_2$')
    plt.savefig('Hist_a2.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,6],density=True,bins=20)
    plt.xlabel('Sigma')
    plt.savefig('Hist_Sigma.png')
    plt.figure()
    plt.hist(chain[Start_Stats:,7],density=True,bins=20)
    plt.xlabel('Degrees of freedom')
    plt.savefig('Hist_d.png')
    
    
    #Plotting Markov Chains
    plt.figure()
    plt.plot(chain[:,0]+StartYear,'*',label='First Break')
    plt.plot(chain[:,1]+StartYear,'r*', label='Second Break')
    plt.xlabel('Samples')
    plt.ylabel('Breaks')
    plt.legend(loc='best',frameon=1)
    plt.savefig('MH_Breaks.png')
    plt.figure()
    plt.plot(chain[:,2],'*')
    plt.xlabel('Samples')
    plt.ylabel('$b$')
    plt.savefig('MH_b.png')
    plt.figure()
    plt.plot(chain[:,3],'*',label='$a_0$')
    plt.xlabel('Samples')
    plt.ylabel('$a_0$')
    plt.savefig('MH_a0.png')
    plt.figure()
    plt.plot(chain[:,4],'*',label='$a_1$')
    plt.xlabel('Samples')
    plt.ylabel('$a_1$')
    plt.savefig('MH_a1.png')
    plt.figure()
    plt.plot(chain[:,5],'*',label='$a_2$')
    plt.xlabel('Samples')
    plt.ylabel('$a_2$')
    plt.savefig('MH_a2.png')
    plt.figure()
    plt.plot(chain[:,6],'*')
    plt.xlabel('Samples')
    plt.ylabel('Sigma')
    plt.savefig('MH_Sigma.png')
    plt.figure()
    plt.plot(chain[:,7],'*')
    plt.xlabel('Samples')
    plt.ylabel('$d$')
    plt.savefig('MH_d.png')
    
###################################################################################    
elif Method=='GVA':  
    #Gaussian Variatonal Approximation
    #Functions needed for method

    #Numerical approximation of the gradient
    def NumGrad8(f,B,P,a1,a2,sig,d,h=0.01):
        a1=np.array([a1])
        a2=np.array([a2])
        G1=(f(np.array([B[0]+h,B[1]]),P,a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G2=(f(np.array([B[0],B[1]+h]),P,a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G3=(f(B,np.array([P[0]+h,P[1]]),a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G4=(f(B,np.array([P[0],P[1]+h]),a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G5=(f(B,P,a1+h,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G6=(f(B,P,a1,a2+h,sig,d)-f(B,P,a1,a2,sig,d))/h
        G7=(f(B,P,a1,a2,sig+h,d)-f(B,P,a1,a2,sig,d))/h
        G8=(f(B,P,a1,a2,sig,d+h)-f(B,P,a1,a2,sig,d))/h
        return np.array([G1,G2,G3,G4,G5,G6,G7,G8])
    
    #Numerical second derivative for GVA where covariance is diagonal
    def NumLfact(f,B,P,a1,a2,sig,d,h=0.01):
        a1=np.array([a1])
        a2=np.array([a2])
        L1=(f(np.array([B[0]+2*h,B[1]]),P,a1,a2,sig,d)-2*f(np.array([B[0]+h,B[1]]),P,a1,a2,sig,d)\
            +f(B,P,a1,a2,sig,d))/(h**2)
        L2=(f(np.array([B[0],B[1]+2*h]),P,a1,a2,sig,d)-2*f(np.array([B[0],B[1]+h]),P,a1,a2,sig,d)\
            +f(B,P,a1,a2,sig,d))/(h**2)
        L3=(f(B,np.array([P[0]+2*h,P[1]]),a1,a2,sig,d)-2*f(B,np.array([P[0]+h,P[1]]),a1,a2,sig,d)\
            +f(B,P,a1,a2,sig,d))/(h**2)
        L4=(f(B,np.array([P[0],P[1]+2*h]),a1,a2,sig,d)-2*f(B,np.array([P[0]+h,P[1]+h]),a1,a2,sig,d)\
            +f(B,P,a1,a2,sig,d))/(h**2)
        L5=(f(B,P,a1+2*h,a2,sig,d)-2*f(B,P,a1+h,a2,sig,d)+f(B,P,a1,a2,sig,d))/(h**2)
        L6=(f(B,P,a1,a2+2*h,sig,d)-2*f(B,P,a1,a2+h,sig,d)+f(B,P,a1,a2,sig,d))/(h**2)
        L7=(f(B,P,a1,a2,sig+2*h,d)-2*f(B,P,a1,a2,sig+h,d)+f(B,P,a1,a2,sig,d))/(h**2)
        L8=(f(B,P,a1,a2,sig,d+2*h)-2*f(B,P,a1+h,a2,sig,d+h)+f(B,P,a1,a2,sig,d))/(h**2)
        return np.array([L1,L2,L3,L4,L5,L6,L7,L8])
     ####################################################

    fct=Log_Post_neg
    #GVA Algorithm with the following input:
    #mu0:Starting point for vector of parameters
    #L0:Starting matrix for Covariance Matrix
    #Nosamples: Number of Samples used in each step of stochasitc gradient descent
    #runs: Number of steps in SGD. This is implemented because an algortihm based on a mimimum error 
    #would take too long to run
    #scale: Vector to scale the steps of SGD for mu
    #scale2: Vector to scale the steps of SGD for L
    
    def GVA(mu0=np.array([10,20,0,0,0,0,1,1]),L0=np.array([-1,-2,-3,-4,-3,-3,-2,-1],float),\
            NoSamples=10,runs=100,scale=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5]),\
            scale2=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5])):
        l=NoSamples
        muG=mu0
        L=L0
        dim=np.sqrt(L.size)
        for i in range(runs):
            #print(i)
            eta=np.random.multivariate_normal(mean=np.array([0,0,0,0,0,0,0,0],float),cov=np.eye(8),size=l)
            Postvec=np.zeros([l,1],float)
            Grad_Post=np.zeros([l,8],float)
            L_Post=np.zeros([l,8],float)
            count=0
            #Computing stochastic approximation for gradient
            for j in range(l):
                Lexp=np.exp(L)
                Input=muG+Lexp@eta[j]
                #print('mu',muG,'L',Lexp,'eta',eta[j],'I',Input)
                B=Input[0:2]
                P=Input[2:4]
                a1=Input[4]
                a2=Input[5]
                sigma=Input[-2]
                d=Input[-1]
                #print('I',Input)
                if sigma>0 and d>0 and B[0]<B[1] and B[0]>1 and B[1]<2013-StartYear: 
                    #Make sure that sigma and d are positive and Breaks are in appropriate range
                    Postvec[j]=fct(B,P,np.array([a1]),np.array([a2]),sigma,d)
                    Grad_Post[j,]=np.transpose(NumGrad8(fct,B,P,a1,a2,sigma,d,h=0.0001))
                    L_Post[j,]=np.transpose(NumLfact(fct,B,P,a1,a2,sigma,d,h=0.00001))
                    count=count+1
            if count>0: #Make sure that indeed valid steps have occured
                ELBO=-1/count*sum(Postvec)+dim/2*np.log(2*np.pi*np.e)+np.sum(L) #Calculating ELBO as reference value
                Grad_ELBO=(-1/count)*np.sum(Grad_Post,axis=0)#Update for mu
                AvgL=(1/count)*np.sum(L_Post,axis=0)
                #print('Avg',AvgL)
                L_Elbo0=-np.exp(2*L[0])*AvgL[0]+1        #Computing values for L matrix update
                L_Elbo1=-np.exp(2*L[1])*AvgL[1]+1
                L_Elbo2=-np.exp(2*L[2])*AvgL[2]+1        #Computing values for L matrix update
                L_Elbo3=-np.exp(2*L[3])*AvgL[3]+1
                L_Elbo4=-np.exp(2*L[4])*AvgL[4]+1        #Computing values for L matrix update
                L_Elbo5=-np.exp(2*L[5])*AvgL[5]+1
                L_Elbo6=-np.exp(2*L[6])*AvgL[6]+1        #Computing values for L matrix update
                L_Elbo7=-np.exp(2*L[7])*AvgL[7]+1
                L_Elbo=np.array([L_Elbo0,L_Elbo1,L_Elbo2,L_Elbo3,L_Elbo4,L_Elbo5,L_Elbo6,L_Elbo7])
                muG=muG+((1/(i+1))*Grad_ELBO*scale) #SGD step
                L=L+(1/(i+1)*L_Elbo*scale2) # SGD step
            else:
                muG=muG
                L=L
            #print('mu',muG,ELBO,'L',L,count)
        return muG,L
    start = time.time()
   
    mu,L=GVA(mu0=np.array([Datapoints/3,2*Datapoints/3,intercept,0.01,0.01,0.01,1,4]),L0=np.array([-1,-1,-2,-3,-3,-3,-2,-2],float),\
            NoSamples=10,runs=50,scale=np.array([5e-1,5e-1,1e-2,1e-5,1e-5,1e-5,1e-4,1e-4]),\
            scale2=np.array([1e-4,1e-5,1e-5,1e-5,1e-5,1e-5,1e-3,1e-7]))
    print('Final mu:',mu)
    Cov1=sp.linalg.expm(np.diag(L))#Covariacne for GVA with diagonal matrix
    Cov=Cov1@np.transpose(Cov1)
    print('Diagonal of Covariance Matrix:',np.diag(Cov))
    
    end = time.time()
    Time=end-start
    print('Runtime:', Time)
    xval=np.array(years)-StartYear
    a0=mu[2]
    a1=mu[3]
    a2=mu[4]
    a3=mu[5]

    breaks1=int(mu[0])
    breaks2=int(mu[1])
    print(a0,a1,a2,a3,breaks1+StartYear,breaks2+StartYear)

    years=np.array(years)
    x1=years[0:breaks1+1]-StartYear
    x2=years[breaks1+1:breaks2+1]-StartYear-breaks1-1
    x3=years[breaks2+1:]-StartYear-breaks2-1
    
    New_reg=np.append(np.append([a0+a1*x1],[a0+a1*(breaks1)+a2*x2]),[a0+a1*(breaks1)+a2*(breaks2-breaks1)+a3*x3])
    plt.plot(years,New_reg, color='red', label='Regression with Breaks')
    plt.scatter([breaks1+StartYear,breaks2+StartYear], [intercept,intercept], s=50, color='orange',label='Breakpoints')
    plt.legend(loc='best',frameon=1)
    plt.savefig('Graph.png')
    
    Res=sum((New_reg-total_temps)**2) #Calculating new residuals
    print('Breakpoints:',breaks1+StartYear,breaks2+StartYear)
    print('Intercept:',a0)
    print('Slopes for different intervals:', a1,a2,a3)
    print('Variance Sigma:',mu[6])
    print('Degrees of Freedom',mu[7])
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
    
    #PLotting Marginal distributions
    plt.figure()
    plt.plot(dvec,dplot)
    plt.xlabel('Degrees of Freedom')
    plt.title('Approximation of Marginal Distribution')
    plt.figure()
    plt.plot(sigmavec,sigmaplot)
    plt.xlabel('Sigma')
    plt.title('Approximation of Marginal Distribution')

#############################################################################    
elif Method=='GVA2':  
    #GVA with full Hessian
    #Functions needed for method
    #Numerical approximation for the Hessian
    def NumHes8(f,P,h=0.01):
        Hes=np.zeros([8,8])
        for i1 in range(8):
                for i2 in range(8):
                    P1=np.array([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7]])
                    P2a=np.array([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7]])
                    P2b=np.array([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7]])
                    P1[i1]=P1[i1]+h
                    P2a[i1]=P2a[i1]+h
                    P1[i2]=P1[i2]+h
                    P2b[i1]=P2b[i1]+h
                    #print(P1,P2a,P2b,P)
                    H1=f(np.array([P1[0],P1[1]]),np.array([P1[2],P1[3]]),\
                         np.array([P1[4]]), np.array([P2a[5]]), P1[-2], P1[-1])
                    H2a=f(np.array([P2a[0],P1[1]]),np.array([P2a[2],P2a[3]]),\
                         np.array([P2a[4]]), np.array([P2a[5]]), P2a[-2], P1[-1])
                    H2b=f(np.array([P2b[0],P1[1]]),np.array([P2b[2],P2b[3]]),\
                         np.array([P2b[4]]), np.array([P2b[5]]), P2b[-2], P2b[-1])
                    H3=f(np.array([P[0],P[1]]),np.array([P[2],P[3]]),\
                         np.array([P[4]]), np.array([P[5]]), P[-2], P[-1])
                    H=(H1-H2a-H2b+H3)/(h**2)
                    Hes[i1,i2]=H
        return Hes

    #Numerical approximation of the gradient
    def NumGrad8(f,B,P,a1,a2,sig,d,h=0.01):
        a1=np.array([a1])
        a2=np.array([a2])
        G1=(f(np.array([B[0]+h,B[1]]),P,a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G2=(f(np.array([B[0],B[1]+h]),P,a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G3=(f(B,np.array([P[0]+h,P[1]]),a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G4=(f(B,np.array([P[0],P[1]+h]),a1,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G5=(f(B,P,a1+h,a2,sig,d)-f(B,P,a1,a2,sig,d))/h
        G6=(f(B,P,a1,a2+h,sig,d)-f(B,P,a1,a2,sig,d))/h
        G7=(f(B,P,a1,a2,sig+h,d)-f(B,P,a1,a2,sig,d))/h
        G8=(f(B,P,a1,a2,sig,d+h)-f(B,P,a1,a2,sig,d))/h
        return np.array([G1,G2,G3,G4,G5,G6,G7,G8])

    fct=Log_Post_neg
    #This function works literally in the same way as GVA above. The only difference is that
    #it uses the actual Hessian to calculate the updates for L. Hence the input for L and scale2 needs to be 
    #matrices instead of vectors. In the update it is therefore imporant to symmetrise the matrix.
    def GVA2(mu0=np.array([10,20,0,0,0,0,1,1]),L0=np.diag([-1,-2,-3,-4,-3,-3,-2,-1]),\
            NoSamples=10,runs=100,scale=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5]),\
            scale2=np.array([1e-4,1e-5,1e-6,1e-7,1e-5,1e-5])):
        l=NoSamples
        muG=mu0
        L=L0
        dim=np.sqrt(L.size)
        for i in range(runs):
            eta=np.random.multivariate_normal(mean=np.array([0,0,0,0,0,0,0,0],float),cov=np.eye(8),size=l)
            Postvec=np.zeros([l,1],float)
            Grad_Post=np.zeros([l,8],float)
            L_Post=np.zeros([l,8,8],float)
            count=0
            #Computing stochastic approximation for gradient
            for j in range(l):
            #print(muG,L,eta[j])
                Input=muG+sp.linalg.expm(L)@eta[j]
                B=Input[0:2]
                P=Input[2:4]
                a1=Input[4]
                a2=Input[5]
                sigma=Input[-2]
                d=Input[-1]
                #print('I:',Input)
                #print(coeffs,sigma,d)
                if sigma>0 and d>0 and B[0]<B[1] and B[0]>1 and B[1]<2013-StartYear: 
                    #Make sure that sigma and d are positive and Breaks are in appropriate range
                    Postvec[j]=fct(B,P,np.array([a1]),np.array([a2]),sigma,d)
                    Grad_Post[j,]=np.transpose(NumGrad8(fct,B,P,a1,a2,sigma,d,h=0.0001))
                    L_Post[j,]=NumHes8(fct,Input,h=0.001)
                    count=count+1
                #print(Grad_Post)
            #print(Grad_Post,1/l*np.sum(Grad_Post,axis=0))
            if count>0:
                ELBO=-1/count*sum(Postvec)+dim/2*np.log(2*np.pi*np.e)+np.sum(L) #Calculating ELBO as reference value
                Grad_ELBO=(-1/count)*np.sum(Grad_Post,axis=0)
                AvgL=(1/count)*np.sum(L_Post,axis=0)
                #print('Avg',AvgL)
                L_Elbo1=sp.linalg.expm(2*L)@AvgL
                L_Elbo=-0.5*(L_Elbo1+np.transpose(L_Elbo1))+np.eye(8) #symmetrise L matrix
                muG=muG+((1/(i+1))*Grad_ELBO*scale) #SGD step
                L=L+(1/(i+1)*L_Elbo*scale2) # SGD step
            else:
                muG=muG
                L=L
            #print(muG,ELBO,L,count)
        return muG,L

    start=time.time()
    mu,L=GVA2(mu0=np.array([Datapoints/3,2*Datapoints/3,intercept,0.01,0.01,0.01,1,4]),\
            L0=np.diag([-1,-1,-2,-3,-3,-3,-2,-2.]),\
            NoSamples=10,runs=25,scale=np.array([5e-1,5e-1,1e-2,1e-5,1e-5,1e-5,1e-4,1e-4]),\
            scale2=1e-7*np.ones([8,8])+1e-7*np.diag(np.ones(8))+np.diag([1e-4,1e-5,1e-5,1e-5,1e-5,1e-5,1e-3,1e-7]))
    print('Final mu:',mu,)
    Cov1=sp.linalg.expm(L)#Covariacne for GVA with diagonal matrix
    Cov=Cov1@np.transpose(Cov1)
    print('Covariance Matrix:',Cov)
    
    end = time.time()
    Time=end-start
    print('Runtime:', Time)
    xval=np.array(years)-StartYear
    a0=mu[2]
    a1=mu[3]
    a2=mu[4]
    a3=mu[5]
    #a3=np.mean(chain[:,4])
    breaks1=int(mu[0])
    breaks2=int(mu[1])
    print(a0,a1,a2,a3,breaks1+StartYear,breaks2+StartYear)
    years=np.array(years)
    x1=years[0:breaks1+1]-StartYear
    x2=years[breaks1+1:breaks2+1]-StartYear-breaks1-1
    x3=years[breaks2+1:]-StartYear-breaks2-1
    
    New_reg=np.append(np.append([a0+a1*x1],[a0+a1*(breaks1)+a2*x2]),[a0+a1*(breaks1)+a2*(breaks2-breaks1)+a3*x3])
    plt.plot(years,New_reg, color='red', label='Regression with Breaks')
    plt.scatter([breaks1+StartYear,breaks2+StartYear], [intercept,intercept], s=50, color='orange',label='Breakpoints')
    plt.legend(loc='best',frameon=1)
    plt.savefig('Graph.png')
    
    Res=sum((New_reg-total_temps)**2) #Calculating new residuals
    print('Breakpoints:',breaks1+StartYear,breaks2+StartYear)
    print('Intercept:',a0)
    print('Slopes for different intervals:', a1,a2,a3)
    print('Variance Sigma:',mu[6])
    print('Degrees of Freedom',mu[7])
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
    
    #PLotting Marginals for Degrees of Freedom and Sigma
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
    #Laplace method for approximation
    #Numerical approximation for the Hessian
    def NumHes8(f,P,h=0.01):
        Hes=np.zeros([8,8])
        for i1 in range(8):
                for i2 in range(8):
                    P1=np.array([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7]])
                    P2a=np.array([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7]])
                    P2b=np.array([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7]])
                    P1[i1]=P1[i1]+h
                    P2a[i1]=P2a[i1]+h
                    P1[i2]=P1[i2]+h
                    P2b[i1]=P2b[i1]+h
                    #print(P1,P2a,P2b,P)
                    H1=f(np.array([P1[0],P1[1]]),np.array([P1[2],P1[3]]),\
                         np.array([P1[4]]), np.array([P2a[5]]), P1[-2], P1[-1])
                    H2a=f(np.array([P2a[0],P1[1]]),np.array([P2a[2],P2a[3]]),\
                         np.array([P2a[4]]), np.array([P2a[5]]), P2a[-2], P1[-1])
                    H2b=f(np.array([P2b[0],P1[1]]),np.array([P2b[2],P2b[3]]),\
                         np.array([P2b[4]]), np.array([P2b[5]]), P2b[-2], P2b[-1])
                    H3=f(np.array([P[0],P[1]]),np.array([P[2],P[3]]),\
                         np.array([P[4]]), np.array([P[5]]), P[-2], P[-1])
                    H=(H1-H2a-H2b+H3)/(h**2)
                    Hes[i1,i2]=H
        return Hes
    
    start=time.time()
    Nopoints2=20000
    Points=np.zeros([Nopoints2,8])
    Base=np.array([0,0,intercept-1,-0.05,-0.05,-0.05,0,0])
    Factor=np.array([Datapoints/2,Datapoints/2,2,0.1,0.1,0.1,3,10])
    Postvec=np.zeros(Nopoints2)
    #Algortihm to find an approximation of the maximum of the Unnormalised Posterior
    for k in range(Nopoints2):
        R=np.random.random(8)
        Point=Base+R*Factor
        Point[1]=Datapoints-Point[1]
        Points[k]=Point
        Postvec[k]=Log_Post(np.array([Point[0],Point[1]]),np.array([Point[2],Point[3]]),\
               np.array([Point[4]]), np.array([Point[5]]), Point[-2], Point[-1])
        

    Mindx=np.argmax(Postvec)
    MaxPoint=Points[Mindx]
    print('Mean:',MaxPoint)
    #Calculating Covariance matrix using the Hessian
    beta=-NumHes8(Log_Post,MaxPoint)
    Cov=np.linalg.inv(beta)
    print('Covariance:', Cov)
    xval=np.array(years)-StartYear
    a0=MaxPoint[2]
    a1=MaxPoint[3]
    a2=MaxPoint[4]
    a3=MaxPoint[5]
    a6=MaxPoint[6]
    a7=MaxPoint[7]
    #a3=np.mean(chain[:,4])
    breaks1=int(MaxPoint[0])
    breaks2=int(MaxPoint[1])

    years=np.array(years)
    x1=years[0:breaks1+1]-StartYear
    x2=years[breaks1+1:breaks2+1]-StartYear-breaks1-1
    x3=years[breaks2+1:]-StartYear-breaks2-1
    
    New_reg=np.append(np.append([a0+a1*x1],[a0+a1*(breaks1)+a2*x2]),[a0+a1*(breaks1)+a2*(breaks2-breaks1)+a3*x3])
    plt.plot(years,New_reg, color='red', label='Regression with Breaks')
    plt.scatter([breaks1+StartYear,breaks2+StartYear], [intercept,intercept], s=50, color='orange',label='Breakpoints')
    plt.legend(loc='best',frameon=1)
    plt.savefig('Graph.png')
    
    Res=sum((New_reg-total_temps)**2) #Calculating new residuals    
    print('Breakpoints:',breaks1+StartYear,breaks2+StartYear)
    print('Intercept:',a0)
    print('Slopes for different intervals:', a1,a2,a3)
    print('Variance Sigma:',a6)
    print('Degrees of Freedom',a7)
    print('Sum of Squares Model:',Res)
    
    muL=np.array([MaxPoint[1],MaxPoint[0]])
    #sigmaL=NumHes3a(LogPost,0.01,MaxPoint[2],muL)
    #sigmaL2=NumHes3a(Post,0.01,MaxPoint[2],muL)
    end=time.time()
    Time=end-start
    print('Runtime:',Time)
           