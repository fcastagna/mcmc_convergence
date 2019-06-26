import numpy as np
import pymc3 as pm
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

obs = [-4.]
niter = 5000
nburn = 2000
nchains = 3
with pm.Model() as model:
    m = pm.Uniform('m', lower=0., upper=10.)
    likelihood = pm.Normal('y_obs', mu=m, sigma=2., observed=obs)
    step_0 = pm.Metropolis(S=np.array([.01]))
    step_1 = pm.Metropolis(S=np.array([10]))
    step_2 = pm.Metropolis(S=np.array([2.5]))
    trace_0 = pm.sample(niter, step=step_0, cores=1, tune=0, chains=nchains, random_seed=123)
    trace_1 = pm.sample(niter, step=step_1, cores=1, tune=0, chains=nchains, random_seed=123)
    trace_2 = pm.sample(niter, step=step_2, cores=1, tune=0, chains=nchains, random_seed=123)
    
# save chains
np.savetxt('chain_1.txt', trace_0.m)
np.savetxt('chain_2.txt', trace_1.m)
np.savetxt('chain_3.txt', trace_2.m)
    
# burn-in
t_0 = trace_0[nburn:]
t_1 = trace_1[nburn:]
t_2 = trace_2[nburn:]

# traceplot
pm.traceplot(trace_0, var_names=['m'], figsize=(16, 6), trace_kwargs={'linewidth': 2}, textsize=20)
plt.axvline(nburn, color='r', linestyle=':')
pm.traceplot(trace_1, var_names=['m'], figsize=(16, 6), trace_kwargs={'linewidth': 2}, textsize=20)
plt.axvline(nburn, color='r', linestyle=':')
pm.traceplot(trace_2, var_names=['m'], figsize=(16, 6), trace_kwargs={'linewidth': 2}, textsize=20)
plt.axvline(nburn, color='r', linestyle=':')

pm.traceplot(trace_0[nburn:nburn+100], var_names=['m'], figsize=(16, 6), textsize=20, trace_kwargs={'linewidth': 2})
pm.traceplot(trace_1[nburn:nburn+100], var_names=['m'], figsize=(16, 6), textsize=20, trace_kwargs={'linewidth': 2})
pm.traceplot(trace_2[nburn:nburn+100], var_names=['m'], figsize=(16, 6), textsize=20, trace_kwargs={'linewidth': 2})

# posterior distribution
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['axes.labelsize'] = 40
pm.plot_posterior(t_2, var_names=['m'], kind='hist', density=True)
def prior(x):
    return st.uniform.pdf(x, 0., 10.) 
def likelihood(x):
    return st.norm.pdf(x, -4., 2.)
def numerator(x):
    return prior(x)*likelihood(x)
def posterior(x):
    return numerator(x)/quad(numerator, 0., 10.)[0]
x = np.arange(0., 10., .001)
plt.plot(x, posterior(x), 'g')
plt.xlim(0, 4)
plt.legend(('94% HPD', 'analytical', 'simulated'), prop={'size': 40})

# summary
pm.summary(t_0)
pm.summary(t_1)
pm.summary(t_2)

# forest plot
pm.forestplot([t_0, t_1, t_2], figsize=(16, 12), textsize=20, markersize=20)

# acceptance rate
print('Model 0: acc_rate = '+str(step_0.accepted/(niter*nchains)))
print('Model 1: acc_rate = '+str(step_1.accepted/(niter*nchains)))
print('Model 2: acc_rate = '+str(step_2.accepted/(niter*nchains)))

# ACF
pm.autocorrplot(t_0, var_names=['m'], combined=True, textsize=20)
pm.autocorrplot(t_1, var_names=['m'], combined=True, textsize=20)
pm.autocorrplot(t_2, var_names=['m'], combined=True, textsize=20)

# ESS
print(pm.effective_n(t_0))
print(pm.effective_n(t_1))
print(pm.effective_n(t_2))

# Gelman-Rubin
print(pm.gelman_rubin(t_0))
print(pm.gelman_rubin(t_1))
print(pm.gelman_rubin(t_2))

# Geweke
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['axes.labelsize'] = 40
gew_0 = pm.geweke(t_0.m, first=.1, last=.5)
plt.scatter(gew_0[:,0], gew_0[:,1])
gew_1 = pm.geweke(t_1.m, first=.1, last=.5)
plt.scatter(gew_1[:,0], gew_1[:,1])
gew_2 = pm.geweke(t_2.m, first=.1, last=.5)
plt.scatter(gew_2[:,0], gew_2[:,1])
plt.xlabel('First iteration')
plt.ylabel('Z-score')
plt.ylim(-4, 4)
plt.axhline(-st.norm.ppf(.95), linestyle='--')
plt.axhline(st.norm.ppf(.95), linestyle='--', label='_nolegend_')
plt.legend(('90% CI', 'Z-scores 0', 'Z-scores 1', 'Z-scores 2'), bbox_to_anchor=(1,1), prop={'size': 25})
