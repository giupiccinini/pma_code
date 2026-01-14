import os
from multiprocessing import Pool
import numpy as np
import random
import emcee

p0 = params + 1.e-6*np.random.randn(ndata['nwalkers'], nparams)
 
with Pool(processes=nthreads) as pool:

	sampler = emcee.EnsembleSampler(ndata['nwalkers_bi'], nparams, log_probability, moves = emcee.moves.DEMove(), pool=pool)
	p0, lp, _ = sampler.run_mcmc(p0, ndata['nburn'], progress=True)

	p0 = p0[np.argmax(lp)]

	p0 = p0 + 1.e-6*np.random.randn(ndata['nwalkers'], nparams)
	sampler.reset()

	sampler = emcee.EnsembleSampler(ndata['nwalkers'], nparams, log_probability, moves = emcee.moves.DEMove(), pool=pool)
	
	count = 0
	chain = np.zeros([nsteps,nwalkers,nparams],dtype=np.float64)
	probability = np.zeros([nsteps,nwalkers],dtype=np.float64)

	for result in sampler.sample(initial_state=p0, iterations=ndata['niter'], store=True, thin_by=1, progress=True):
		chain[count,:,:] = result[0]
		probability[count,:] = result[1]
		
		count += 1
		
