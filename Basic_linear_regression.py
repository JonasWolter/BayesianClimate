
#Importiing required libraries
import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#####################################################################
#Description: 
#This Program fits a basic basyesian linear regression (Gaussian Prior + Gaussian Noise)
#to the temeprature on different locations in the world. The location (countries) can be chosenn 
#with the Location parameter.  Three different approximations
#have been implemented wich can be chosen through the Method parameter. 
#Parameteres for different models are set in appropriate section. Different location may require 
#a change in the stepsize for MH or the scale for GVA.
#
#As the Output the script first gives the parameters of basic linear regression and the sum of least squares.
#Secondly approximates of the estimated parameters are given. These are either the estimates for the
#mean for the Gaussian for GVA and Laplace, or the mean value of the Markov chain after the burn-in period for 
#Metropolis Hasting. Furthermore the sum of squares of the estimate is printed. For MH the acceptance ratio is given as well.
#Lastly we see the runtime to get an idea of the performance of the used algortihm.
#In addition some graphs are plotted which should be self-explanatory.

Location='Global' #Selecting Location for which Data should be analysed, 'Global' for whole earth 
Method='GVA2'#Choose bettwen Laplace, Metropolis Hasting (MH) and Gaussian Variational Approximation (GVA) 
            #with a diagonal Hessian or GVA2 with the full Hessian. In general GVA2 is preferred but it 
            #takes longer to run. The default parameters yield acceptable results in a few minutes.
            #For more accurate results a longer time is required, which can be achieved by channging the parameters.  
            
####################################################################### 
            
plt.close('all')
import warnings
warnings.filterwarnings('ignore')



#Function needed in code
def sum_handles2(handles_list):
    def aux(x1,x2):
        temp=0
        for f in handles_list:
            temp=temp+f(x1,x2)
        return temp
    return aux

def multi_Gamma(mu,S,a,b):
    d=(np.size(S))**0.5
    def mGaux(beta,Theta):
        return (b**a/sp.special.gamma(a))*(np.linalg.det(S)**(1/2))/((2*np.pi)**(d/2))* \
                np.exp(-b*beta-0.5*beta*np.transpose(Theta-mu)@S@(Theta-mu))
    return mGaux

####################################################               

if Location=='Global':
    StartYear=1830
    Datapoints=180
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
string='Average temperature in {}'.format(Location)
plt.title(string)

#Doing Basic linear Regression
y=np.array(total_temps[0:Datapoints])
x=np.array(years[0:Datapoints])-(StartYear+1)
x_mean=np.mean(x)
y_mean=np.mean(y)

slope=(x-x_mean).dot(y-y_mean)/((x-x_mean).dot(x-x_mean))
intercept=y_mean-slope*x_mean
print('Basic linear regression:')
print('Intercept:',intercept,'Slope:', slope)

xplot=(np.array(years)-(StartYear+1))
LinRegr_Est=slope*(np.array(years)-(StartYear+1))+intercept
plt.plot(years,LinRegr_Est,label='Basic Linear Regression', color='green')
Res0=sum((LinRegr_Est-total_temps)**2) #Calculating Residual
print('Sum of Squares standard linear regression:', Res0)


#Basic Bayesian Linear Regression
mu=np.array([-10,3])
S=np.array([[0.1,0],[0,1]])
a=1
b=1
Sigma=1

#Define Prior
#In the basic case of a Gaussian prior with Gaussian Noise we get a multi-variate Gamma 
#distribution as a Prior as discussed in the course
Prior=multi_Gamma(mu,S,a,b)



def LogLikli(x,y):
    def LogLikli_dens(beta,Theta):
        xvec=np.array([1,x])
        return  np.log(beta**(1/2))+(-beta/2*(y-Theta.dot(xvec))**2)
    return LogLikli_dens



Liklilist=[]
count=0

for xi,yi in zip(x,y):#zip(years[0:Datapoints],total_temps[0:Datapoints]):
    #print(xi,yi)
    count=count+1
    L1=LogLikli(xi,yi)
    Liklilist.append(L1)
    
Lik=sum_handles2(Liklilist)   

#Calculate Posterior
def Post(beta,Theta):
    return Prior(beta,Theta)*np.exp(Lik(beta,Theta))

def LogPost(beta,Theta):
    return np.log(Prior(beta,Theta))+Lik(beta,Theta)


def LogPost_neg(beta,Theta):
    return -(np.log(Prior(beta,Theta))+Lik(beta,Theta))


# Printing what you like is important for success
for i in range(1):
    print("Eier~Eier~Eier")



#################################################################
if Method=='MH':
    #Metropolis Hastigns
    def MH_Gaus(LogPost, stepsize=np.array([1,1,0.0001]), No_samples=1000,init=[4,intercept,0]):
        chain=np.zeros([No_samples,3])
        count=0
        chain[0,:]=init
        for i in range(No_samples-1):
            curr=chain[i,:]
            prop=curr+stepsize*np.random.multivariate_normal([0,0,0],np.eye(3))
            
            prop[0]=prop[0]
            Check=min(1,np.exp(LogPost(prop[0],prop[1:])-LogPost(curr[0],curr[1:])))
            U=np.random.uniform(0,1,1)
            #print(prop,Log_Post(prop[0:2],prop[2],np.floor(prop[3])),Check,U)
            if U <Check:
                    chain[i+1,:]=prop
                    count=count+1
            else:
                    chain[i+1,:]=curr
        Ratio=count/No_samples
        return chain,Ratio
    
    start=time.time()
    Start_Stats=1000
    runs=10000
    chain,Ratio=MH_Gaus(LogPost, stepsize=np.array([2,0.01,0.001]), No_samples=runs,init=[4,intercept,0])
    end = time.time()
    Time=end-start
    print('Runtime:',Time,'Sekunden')
    print('Acceptance Ratio Markov Chain:',Ratio)
    
    #Getting values for plotting results
    a0=np.mean(chain[Start_Stats:,1])
    a1=np.mean(chain[Start_Stats:,2])
    a2=np.mean(chain[Start_Stats:,0])
    
    print('Slope:',a1)
    print('Intercept:',a0)
    print('Degrees of Freedom:',a2)

    
    years_a=np.array(years)-StartYear
    
    BayLinRegr2=a1*years_a+a0
    
    plt.plot(years,BayLinRegr2,label='Bayesian Linear Regression 2', color='orange')
    legend = plt.legend(loc='best',frameon=1)
    Res=sum((BayLinRegr2-total_temps)**2) #Calculating new residuals
    print('Sum of Squares Model:', Res)
    
    plt.figure()
    plt.hist(chain[:,0],density=True,bins=20)
    plt.xlabel('Degrees of Freedom')
    plt.figure()
    plt.hist(chain[:,1],bins=20)
    plt.xlabel('Intercept')
    plt.figure()
    plt.hist(chain[:,2],density=True,bins=20)
    plt.xlabel('Slope')
    
    plt.figure()
    plt.plot(chain[:,0],'*')
    plt.xlabel('Samples')
    plt.ylabel('Degrees of Freedom')
    plt.figure()
    plt.plot(chain[:,1],'*')
    plt.xlabel('Samples')
    plt.ylabel('Intercept')
    plt.figure()
    plt.plot(chain[:,2],'*')
    plt.xlabel('Samples')
    plt.ylabel('Slope')
    
##############################################
elif Method=='GVA': 
    #Numerical approximation of the gradient
    def NumGrad(f,d,B,h=0.01):
        G1=(f(d+h,np.array([B[0],B[1]]))-f(d,B))/h
        G2=(f(d,np.array([B[0]+h,B[1]]))-f(d,B))/h
        G3=(f(d,np.array([B[0],B[1]+h]))-f(d,B))/h
        return np.array([G1,G2,G3])
    
        #Numerical second derivative for GVA where covariance is diagonal
    def NumLfact(f,d,B,h=0.01):
        L1=(f(d+2*h,np.array([B[0],B[1]]))-2*f(d+h,np.array([B[0],B[1]]))\
            +f(d,B))/(h**2)
        L2=(f(d,np.array([B[0]+2*h,B[1]]))-2*f(d,np.array([B[0]+h,B[1]]))\
            +f(d,B))/(h**2)
        L3=(f(d,np.array([B[0],B[1]+2*h]))-2*f(d,np.array([B[0],B[1]+h]))\
            +f(d,B))/(h**2)
        return np.array([L1,L2,L3])
    
    #Gaussian Variatonal Approximation
    fct=LogPost_neg
    def GVA(mu0=np.array([1,intercept,0]),L0=np.array([-1,-1,-1],float),\
            NoSamples=10,runs=100,scale=np.array([1e-4,1e-5,1e-5]),\
            scale2=np.array([1e-4,1e-5,1e-6])):
        l=NoSamples
        muG=mu0
        L=L0
        dim=np.sqrt(L.size)
        for i in range(runs):
            dold=muG[0]
            eta=np.random.multivariate_normal(mean=np.array([0,0,0],float),cov=np.eye(3),size=l)
            Postvec=np.zeros([l,1],float)
            Grad_Post=np.zeros([l,3],float)
            L_Post=np.zeros([l,3],float)
            count=0
            #Computing stochastic approximation for gradient
            for j in range(l):
                Lexp=np.exp(L)
                Input=muG+Lexp@eta[j]
                #print('mu',muG,'L',Lexp,'eta',eta[j],'I',Input)
                coeffs=Input[1:]
                #print(coeffs)
                d=round(Input[0])
                #print(coeffs,d)
                if d>0: #Make sure that sigma and d are positive
                    Postvec[j]=fct(d,coeffs)
                    Grad_Post[j,]=NumGrad(fct,d,coeffs,h=0.0001)
                    L_Post[j,]=NumLfact(fct,d,coeffs,h=0.00001)
                    #print(Grad_Post)
                    count=count+1
                if count>0:
                    ELBO=-1/count*sum(Postvec)+dim/2*np.log(2*np.pi*np.e)+np.sum(L)
                    Grad_ELBO=(-1/count)*np.sum(Grad_Post,axis=0)#Update for mu
                    #print(Grad_ELBO)
                    AvgL=(1/count)*np.sum(L_Post,axis=0)
                    #print('Avg',AvgL)
                    L_Elbo0=-np.exp(2*L[0])*AvgL[0]+1        #Computing values for L matrix update
                    L_Elbo1=-np.exp(2*L[1])*AvgL[1]+1
                    L_Elbo2=-np.exp(2*L[2])*AvgL[2]+1        #Computing values for L matrix update
                    L_Elbo=np.array([L_Elbo0,L_Elbo1,L_Elbo2])
                    muG=muG+((1/(i+1))*Grad_ELBO*scale) #SGD step
                    L=L+(1/(i+1)*L_Elbo*scale2) # SGD step
                else:
                    muG=muG
                    L=L
                muG[0]=round(muG[0]) #make degrees of freedom integers
                if muG[0]<1:
                    muG[0]=dold #make sure that degrees of freedom are positive
            #print('mu',muG,ELBO,'L',L)
        return muG,L
    start = time.time()
    mu,L=GVA(mu0=np.array([10,intercept,0]),L0=np.array([-2,-3,-3],float),\
            NoSamples=10,runs=200,scale=np.array([1e-4,1e-4,1e-7]),\
            scale2=np.array([1e-1,1e-1,1e-4]))
    print('Final slope:',mu[2])
    print('Final intercept:', mu[1])
    print('Final degree of freedom:', round(mu[0]))
    Cov1=sp.linalg.expm(np.diag(L))#Covariacne for GVA with diagonal matrix
    Cov=Cov1@np.transpose(Cov1)
    print('Cov:',Cov)
    
    end = time.time()
    Time=end-start
    print('Runtime:', Time)
    xval=np.array(years)-StartYear
    Bays_linear=mu[1]+mu[2]*xval
    plt.plot(years,Bays_linear, label='Bayesian Polynomial Regression', color='orange')
    plt.legend(loc='best',frameon=1)
    Res=sum((Bays_linear-total_temps)**2) #Calculating new residuals
    print('Sum of Squares Model:', Res)
    #Defining Standard Gaussian
    def uni_Gaus(mean,sigma):
        def uni_Gaus_unnorm_dens(x):
            return 1/(np.sqrt(2*sigma**2*np.pi))*np.exp(-(x-mean)**2/(2*sigma**2))
        return uni_Gaus_unnorm_dens
    
    #Plotting distribution of sigma and d
    plotN=100
    Distr_inter=uni_Gaus(mu[1],Cov[(1,1)])
    intervec=np.zeros([plotN,])
    interplot=np.zeros([plotN,])
    Distr_slope=uni_Gaus(mu[-1],Cov[(-1,-1)])
    slopevec=np.zeros([plotN,])
    slopeplot=np.zeros([plotN,])
    for i in range(plotN):
        slope=mu[-1]-4*Cov[(-1,-1)]+i/plotN*8*Cov[(-1,-1)]
        slopevec[i]=slope
        slopeplot[i]=Distr_slope(slope)
        inter=mu[-2]-4*Cov[(-2,-2)]+i/plotN*8*Cov[(-2,-2)]
        intervec[i]=inter
        interplot[i]=Distr_inter(inter)
      
    plt.figure()
    plt.plot(slopevec,slopeplot)
    plt.xlabel('Slope')
    plt.title('Approximation of Marginal Distribution')
    plt.figure()
    plt.plot(intervec,interplot)
    plt.xlabel('Interception')
    plt.title('Approximation of Marginal Distribution')
   
##############################################
elif Method=='GVA2': 
    #Numerical approximation of the gradient
    def NumGrad(f,d,B,h=0.01):
        G1=(f(d+h,np.array([B[0],B[1]]))-f(d,B))/h
        G2=(f(d,np.array([B[0]+h,B[1]]))-f(d,B))/h
        G3=(f(d,np.array([B[0],B[1]+h]))-f(d,B))/h
        return np.array([G1,G2,G3])
    
    #Numerical Approximation of Hessian
    def NumHes3a(f,Q,P,h):
        Pa=np.array([P[0]+h,P[1]])
        Pb=np.array([P[0],P[1]+h])
        Pc=np.array([P[0]+h,P[1]+h])
        Pd=np.array([P[0]+2*h,P[1]])
        Pe=np.array([P[0],P[1]+2*h])
        H11=(f(Q+2*h,P)-2*f(Q+h,P)+f(Q,P))/(h**2)
        H12=(f(Q+h,Pa)-f(Q,Pa)-f(Q+h,P)+f(Q,P))/(h**2)
        H13=(f(Q+h,Pb)-f(Q,Pb)-f(Q+h,P)+f(Q,P))/(h**2)
        H22=(f(Q,Pd)-2*f(Q,Pa)+f(Q,P))/(h**2)
        H23=(f(Q,Pc)-f(Q,Pa)-f(Q,Pb)+f(Q,P))/(h**2)
        H33=(f(Q,Pe)-2*f(Q,Pb)+f(Q,P))/(h**2) 
        H=np.array([[H11,H12,H13],[H12,H22,H23],[H13,H23,H33]])
        return H

    
    #Gaussian Variatonal Approximation
    fct=LogPost_neg
    def GVA2(mu0=np.array([1,intercept,0]),L0=np.diag([-1,-1,-1]),\
            NoSamples=10,runs=100,scale=np.array([1e-3,1e-5,1e-5]),\
            scale2=np.array([1e-4,1e-5,1e-6])):
        l=NoSamples
        muG=mu0
        L=L0
        dim=np.sqrt(L.size)
        for i in range(runs):
            dold=muG[0]
            eta=np.random.multivariate_normal(mean=np.array([0,0,0],float),cov=np.eye(3),size=l)
            Postvec=np.zeros([l,1],float)
            Grad_Post=np.zeros([l,3],float)
            L_Post=np.zeros([l,3,3],float)
            count=0
            #Computing stochastic approximation for gradient
            for j in range(l):
                #print(L)
                Lexp=sp.linalg.expm(L)
                Input=muG+Lexp@eta[j]
                #print('mu',muG,'L',Lexp,'eta',eta[j],'I',Input)
                coeffs=Input[1:]
                #print(coeffs)
                d=round(Input[0])
                #print(coeffs,d)
                if d>0: #Make sure that sigma and d are positive
                    Postvec[j]=fct(d,coeffs)
                    Grad_Post[j,]=NumGrad(fct,d,coeffs,h=0.0001)
                    L_Post[j,]=NumHes3a(fct,d,coeffs,h=0.00001)
                    #print(Grad_Post)
                    count=count+1
                if count>0:
                    ELBO=-1/count*sum(Postvec)+dim/2*np.log(2*np.pi*np.e)+np.sum(L)
                    Grad_ELBO=(-1/count)*np.sum(Grad_Post,axis=0)#Update for mu
                    #print(Grad_ELBO)
                    AvgL=(1/count)*np.sum(L_Post,axis=0)
                    #print('Avg',AvgL)
                    L_Elbo1=sp.linalg.expm(2*L)@AvgL
                    L_Elbo=-0.5*(L_Elbo1+np.transpose(L_Elbo1))+np.eye(3) #symmetrise L matrix])
                    muG=muG+((1/(i+1))*Grad_ELBO*scale) #SGD step
                    L=L+(1/(i+1)*L_Elbo*scale2) # SGD step
                else:
                    muG=muG
                    L=L
                muG[0]=muG[0] #make degrees of freedom integers
                if muG[0]<1:
                    muG[0]=dold #make sure that degrees of freedom are positive
            #print('mu',muG,ELBO,'L',L)
        return muG,L
    start = time.time()
    mu,L=GVA2(mu0=np.array([10,intercept,0]),L0=np.diag([-2,-3,-3]),\
            NoSamples=10,runs=200,scale=np.array([1e-4,1e-4,1e-7]),\
            scale2=1e-7*np.ones([3,3])+1e-7*np.diag(np.ones(3))+np.diag([1e-2,1e-1,1e-4]))
    print('Final slope:',mu[2])
    print('Final intercept:', mu[1])
    print('Final degree of freedom:', mu[0])
    Cov1=sp.linalg.expm(L)#Covariacne for GVA with diagonal matrix
    Cov=Cov1@np.transpose(Cov1)
    print('Cov:',Cov)
    
    end = time.time()
    Time=end-start
    print('Runtime:', Time)
    xval=np.array(years)-StartYear
    Bays_linear=mu[1]+mu[2]*xval
    plt.plot(years,Bays_linear, label='Bayesian Polynomial Regression', color='orange')
    plt.legend(loc='best',frameon=1)
    Res=sum((Bays_linear-total_temps)**2) #Calculating new residuals
    print('Sum of Squares Model:', Res)
    
    #Defining Standard Gaussian
    def uni_Gaus(mean,sigma):
        def uni_Gaus_unnorm_dens(x):
            return 1/(np.sqrt(2*sigma**2*np.pi))*np.exp(-(x-mean)**2/(2*sigma**2))
        return uni_Gaus_unnorm_dens
    
    #Plotting distribution of sigma and d
    plotN=100
    Distr_inter=uni_Gaus(mu[1],Cov[(1,1)])
    intervec=np.zeros([plotN,])
    interplot=np.zeros([plotN,])
    Distr_slope=uni_Gaus(mu[-1],Cov[(-1,-1)])
    slopevec=np.zeros([plotN,])
    slopeplot=np.zeros([plotN,])
    for i in range(plotN):
        slope=mu[-1]-4*Cov[(-1,-1)]+i/plotN*8*Cov[(-1,-1)]
        slopevec[i]=slope
        slopeplot[i]=Distr_slope(slope)
        inter=mu[-2]-4*Cov[(-2,-2)]+i/plotN*8*Cov[(-2,-2)]
        intervec[i]=inter
        interplot[i]=Distr_inter(inter)
      
    plt.figure()
    plt.plot(slopevec,slopeplot)
    plt.xlabel('Slope')
    plt.title('Approximation of Marginal Distribution')
    plt.figure()
    plt.plot(intervec,interplot)
    plt.xlabel('Interception')
    plt.title('Approximation of Marginal Distribution')
   

#############################################
if Method=='Laplace':    
    #Laplace Approximation
    #Numerical Approximation of Hessian
    def NumHes3a(f,Q,P,h):
        Pa=np.array([P[0]+h,P[1]])
        Pb=np.array([P[0],P[1]+h])
        Pc=np.array([P[0]+h,P[1]+h])
        Pd=np.array([P[0]+2*h,P[1]])
        Pe=np.array([P[0],P[1]+2*h])
        H11=(f(Q+2*h,P)-2*f(Q+h,P)+f(Q,P))/(h**2)
        H12=(f(Q+h,Pa)-f(Q,Pa)-f(Q+h,P)+f(Q,P))/(h**2)
        H13=(f(Q+h,Pb)-f(Q,Pb)-f(Q+h,P)+f(Q,P))/(h**2)
        H22=(f(Q,Pd)-2*f(Q,Pa)+f(Q,P))/(h**2)
        H23=(f(Q,Pc)-f(Q,Pa)-f(Q,Pb)+f(Q,P))/(h**2)
        H33=(f(Q,Pe)-2*f(Q,Pb)+f(Q,P))/(h**2)
        H=np.array([[H11,H12,H13],[H12,H22,H23],[H13,H23,H33]])
        return H
     
    start=time.time()
    Nopoints2=3000
    Points=np.zeros([Nopoints2,3])
    Base=np.array([slope-0.1,intercept-1,0])
    Factor=np.array([0.2,2,20])
    Postvec2=np.zeros(Nopoints2)
    for k in range(Nopoints2):
        R=np.random.random(3)
        Point=Base+R*Factor
        Point[2]=Point[2]
        Points[k]=Point
        Postvec2[k]=Post(Point[2],np.array([Point[1],Point[0]]))

    Mindx2=np.argmax(Postvec2)
    MaxPoint=Points[Mindx2]
    print('Slope:',MaxPoint[0])
    print('Intercept:',MaxPoint[1])
    print('Degrees of Freedom:',MaxPoint[2])
        
    muL=np.array([MaxPoint[1],MaxPoint[0]])
    sigmaL=NumHes3a(LogPost,MaxPoint[2],muL,h=0.01)
    Cov=np.linalg.inv(sigmaL)
    print('Covariance:',Cov)
    end=time.time()
    Time=end-start
    print('Runtime:',Time)
    
    
    years_a=np.array(years)-StartYear
    
    BayLinRegr2=MaxPoint[0]*years_a+MaxPoint[1]
    
    plt.plot(years,BayLinRegr2,label='Bayesian Linear Regression 2', color='orange')
    legend = plt.legend(loc='best',frameon=1)
    Res=sum((BayLinRegr2-total_temps)**2) #Calculating new residuals
    print('Sum of Squares Model:', Res)
    #Plotting Posterior
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Points[:,0],Points[:,1],Postvec2,c=None, depthshade=True)
    
    ax.set_xlabel('gradient')
    ax.set_ylabel('constant')
    ax.set_zlabel('Unnormalised Posterior')
    ax.set_title('Approximation of Gaussian')
    
    Nopoints=5
    avec=np.zeros(Nopoints**2)
    bvec=np.zeros(Nopoints**2)
    Postvec=np.zeros(Nopoints**2)
    n=Nopoints
    for k1 in range(Nopoints):
        a=-0.01+k1/Nopoints*0.02
        for k2 in range(Nopoints):
            indx=Nopoints*k1+k2
            b=-6+k2/Nopoints*2
            avec[indx]=a
            bvec[indx]=b
            Postvec[indx]=Post(MaxPoint[2],np.array([b,a]))
            
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
    
    Nopoints=50
    avec=np.zeros(Nopoints**2)
    bvec=np.zeros(Nopoints**2)
    Postvec=np.zeros(Nopoints**2)
    
    for k1 in range(Nopoints):
        a=0.006+0.0065*k1/Nopoints
        for k2 in range(Nopoints):
            indx=Nopoints*k1+k2
            b=7.3+0.9*k2/Nopoints
            avec[indx]=a
            bvec[indx]=b
            Postvec[indx]=Post(3,np.array([b,a]))
    
    
    
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
