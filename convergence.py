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
    likelihood = pm.Normal('y_obs', mu=m, sd=2., observed=obs)
    step_1 = pm.Metropolis(S=np.array([.01]))
    step_2 = pm.Metropolis(S=np.array([10]))
    step_3 = pm.Metropolis(S=np.array([2.5]))
    trace_1 = pm.sample(niter, step=step_1, cores=1, tune=0, chains=nchains, random_seed=123)
    trace_2 = pm.sample(niter, step=step_2, cores=1, tune=0, chains=nchains, random_seed=123)
    trace_3 = pm.sample(niter, step=step_3, cores=1, tune=0, chains=nchains, random_seed=123)

# burn-in
t_1 = trace_1[nburn:]
t_2 = trace_2[nburn:]
t_3 = trace_3[nburn:]

# traceplot
pm.traceplot(trace_1[:])
plt.axvline(nburn, color='r', linestyle=':'); plt.ylim(0, 7)
pm.traceplot(trace_2[:])
plt.axvline(nburn, color='r', linestyle=':'); plt.ylim(0, 7)
pm.traceplot(trace_3[:])
plt.axvline(nburn, color='r', linestyle=':'); plt.ylim(0, 7)

pm.traceplot(trace_1[nburn:nburn+100])
pm.traceplot(trace_2[nburn:nburn+100])
pm.traceplot(trace_3[nburn:nburn+100])

# posterior distribution
pm.plot_posterior(t_2, density=True)
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
plt.legend(('94% HPD', 'analytical', 'simulated'))

# summary
pm.summary(t_1)
pm.summary(t_2)
pm.summary(t_3)

# forest plot
pm.forestplot([t_1, t_2, t_3])

# acceptance rate
print(step_1.accepted/(niter*nchains))
print(step_2.accepted/(niter*nchains))
print(step_3.accepted/(niter*nchains))

# ACF
pm.autocorrplot(t_1)
pm.autocorrplot(t_2)
pm.autocorrplot(t_3)

# ESS
print(pm.effective_n(t_1))
print(pm.effective_n(t_2))
print(pm.effective_n(t_3))

# Gelman-Rubin
print(pm.gelman_rubin(t_1))
print(pm.gelman_rubin(t_2))
print(pm.gelman_rubin(t_3))

# Geweke
gew_1 = pm.geweke(t_1.m, first=.1, last=.5)
plt.plot(gew_1[:,0], gew_1[:,1], '.')
gew_2 = pm.geweke(t_2.m, first=.1, last=.5)
plt.plot(gew_2[:,0], gew_2[:,1], '.')
gew_3 = pm.geweke(t_3.m, first=.1, last=.5)
plt.plot(gew_3[:,0], gew_3[:,1], '.')
plt.xlabel('First iteration')
plt.ylabel('Z-score')
plt.ylim(-4, 4)
plt.axhline(-st.norm.ppf(.95), linestyle='--')
plt.axhline(st.norm.ppf(.95), linestyle='--')
plt.legend(('Z-scores 1', 'Z-scores 2', 'Z-scores 3', '90% CI'), bbox_to_anchor=(1,.65))
