import numpy as np
import pandas as pd

# Simulate multivariate normals, and attempt to accurately estimate 
# covariance in the face of systematic mean

nRlz = 1000  # Number of realizations
K = 10       # Number of teams/pipelines
N = 100      # Number of voxels

mu0 = 2  # 2, or try 0 for the complete null case
RemovePipelineMean = 1  # As usually done for covariance
RemoveSpatialMean = 0   # Not usually done

# RemovePipelineMean = 1 
# RemoveSpatialMean = 0
#   True        MC       Est    EstBaT    EstBaE
#   0.82  0.813831  4.817087  0.817087 -0.008248

# RemovePipelineMean = 1
# RemoveSpatialMean = 1
#   True        MC       Est    EstBaT    EstBaE
#   0.82  0.81165  5.134781e-21    -4.0 -4.816477

# RemovePipelineMean = 0
# RemoveSpatialMean = 0
#   True        MC       Est    EstBaT    EstBaE
#   0.82  0.806143  4.819382  0.819382  5.657697e-18


# RemovePipelineMean = 0
# RemoveSpatialMean = 1
#   True        MC       Est    EstBaT    EstBaE
#   0.82  0.810905  7.063794e-20    -4.0 -4.827623


mu = mu0 * np.concatenate((-np.ones(N//2), np.ones(N//2)))
rho = 0.8
Q = np.eye(K) * (1 - rho) + rho * np.ones((K, K))

SE2mc, SE2est, SE2estBaT, SE2estBaE = [], [], [], []
SE2tru = np.sum(Q) / K ** 2  # 1 Q 1' / K^2

for i in range(nRlz):
    # generate mu_i = N(0, Q)
    Y = np.random.multivariate_normal(np.zeros(K), Q, N).T
    Y = Y + mu
    # estimated mean computed
    muhat = np.mean(Y, axis=0)
    e = Y.copy()
    if RemovePipelineMean:
        e = e - np.mean(e, axis=1, keepdims=True)  # Usual centering before computing covariance
    if RemoveSpatialMean:
        e = e - np.mean(e, axis=0)  # Removing spatially-varying mean
    S = np.dot(e, e.T) / N
    SbaTru = S - np.sum(mu ** 2) / N  # Adjust for true mean
    SbaEst = S - np.sum(muhat ** 2) / N  # Adjust for estimated mean

    SE2est.append(np.sum(S) / K ** 2)  # 1 Qhat 1' / K^2
    SE2estBaT.append(np.sum(SbaTru) / K ** 2)
    SE2estBaE.append(np.sum(SbaEst) / K ** 2)

    # Variance about true mu; I argue that this is the relevant quantity for
    # computing uncertainty in muhat at each voxel
    SE2mc.append(np.var(muhat - mu, axis=0))

muSE = pd.DataFrame({
    'True': [SE2tru],
    'MC': [np.mean(SE2mc)],
    'Est': [np.mean(SE2est)],
    'EstBaT': [np.mean(SE2estBaT)],
    'EstBaE': [np.mean(SE2estBaE)]
})


"""
In summary:

Usual, naive computation of covariance S gives unbiased estimate of Q (and hence 1Q1'/K^2) *only* if mu=0
If mu<>0 nothing works... 

-naive estimate S results in over-estimates of 1Q1'/K^2
-removing the voxel-wise means computing S gives 1S1'/K^2 = 0
-an oracle bias correction of S - sum(mu^2)/N *does* work, but requires true mu, and
-a practical bias correction of S - sum(muhat^2)/N over corrects, in fact, over-corrects by 1Q1'/K^2 giving another 0 estimate 

And how does this fit in with NARPS consensus analysis?  
It shows that it is *absolutely* *crucial* not to voxel-wise centre the data. 
Q must be estimated by the original data; any image-wise pipeline differences 
are ignored by the covariance calculation that first enters data image-wise.  
But this also means that Q will necessarily estimate the sum of null 
interpipeline covariance and mean square of mu.  
Under the null, this is fine, but it does mean that under the alternative 
any standard errors will be inflated.
"""


