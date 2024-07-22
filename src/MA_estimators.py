import numpy
import scipy
from sklearn.preprocessing import StandardScaler



####################################
##### META ANALYSIS ESTIMATORS #####
####################################

#########
##### Conventional MA models #####
#########

def Average(contrast_estimates):
    K = contrast_estimates.shape[0]
    intuitive_solution = numpy.sum(contrast_estimates, 0)/K
    T_map = intuitive_solution.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights

def Stouffer(contrast_estimates):
    # compute meta-analytic statistics
    K = contrast_estimates.shape[0] 
    # final stats is mean(Zj)/sqrt(1/k))
    T_map = numpy.mean(contrast_estimates, 0)/(numpy.sqrt(1/K)) # team wise
    T_map = T_map.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights

#########
##### SDMA MA models #####
#########

def SDMA_Stouffer(contrast_estimates):
    K = contrast_estimates.shape[0]
    ones = numpy.ones((K, 1))
    Q = numpy.corrcoef(contrast_estimates)
    attenuated_variance = ones.T.dot(Q).dot(ones) / K**2
    # compute meta-analytic statistics
    T_map = numpy.mean(contrast_estimates, 0)/numpy.sqrt(attenuated_variance)
    T_map = T_map.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights

def Consensus_SDMA_Stouffer(contrast_estimates):
    K = contrast_estimates.shape[0]
    ones = numpy.ones((K, 1))
    Q = numpy.corrcoef(contrast_estimates)
    attenuated_variance = ones.T.dot(Q).dot(ones) / K**2
    consensus_mean = numpy.mean(contrast_estimates, 1).sum() / K # scalar
    # T  =  mean(y,0)/s-hat-2
    # use diag to get s_hat2 for each variable
    T_map = (numpy.mean(contrast_estimates, 0) - consensus_mean
      )/numpy.sqrt(attenuated_variance) + consensus_mean
    T_map = T_map.reshape(-1)
    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights

def Consensus_Average(contrast_estimates):
    K = contrast_estimates.shape[0] # shape contrast_estimates = K*J
    # compute a standardized map Z∗ for mean pipeline z_mean
    scaler = StandardScaler()
    consensus_var = numpy.var(contrast_estimates, 1).sum() / K 
    consensus_std = numpy.sqrt(consensus_var) # SIGMA C
    consensus_mean = numpy.mean(contrast_estimates, 1).sum() / K # MU C 
    Z_star_consensus = (scaler.fit_transform(numpy.mean(contrast_estimates, 0).reshape(-1, 1))) * consensus_std + consensus_mean
    T_map = Z_star_consensus.reshape(-1)
    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights

def SDMA_GLS(contrast_estimates):
    K = contrast_estimates.shape[0]
    Q0 = numpy.corrcoef(contrast_estimates)
    Q = Q0.copy()
    Q_inv = numpy.linalg.inv(Q)
    ones = numpy.ones((K, 1))
    top = ones.T.dot(Q_inv).dot(contrast_estimates)
    down = ones.T.dot(Q_inv).dot(ones)
    T_map = top/numpy.sqrt(down)
    T_map = T_map.reshape(-1)
    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = (ones.T.dot(Q_inv).dot(ones))**(-1/2) * numpy.sum(Q_inv, axis=1) 
    return T_map, p_values, weights


def Consensus_SDMA_GLS(contrast_estimates):
    # compute GLS Stouffer first
    K = contrast_estimates.shape[0]
    Q0 = numpy.corrcoef(contrast_estimates)
    Q = Q0.copy()
    Q_inv = numpy.linalg.inv(Q)
    ones = numpy.ones((K, 1))
    GLS_Stouffer_mean = (ones.T.dot(Q_inv).dot(contrast_estimates)) / (ones.T.dot(Q_inv).dot(ones))
    # then compute the consensus GLS Stouffer
    consensus_mean = numpy.mean(contrast_estimates, 1).sum() / K # scalar
    top = GLS_Stouffer_mean - consensus_mean
    down = ones.T.dot(Q_inv).dot(ones)
    consensus_GLS_Stouffer = top/numpy.sqrt(down**-1) + consensus_mean
    T_map = consensus_GLS_Stouffer.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = ones.T.dot(Q_inv)
    return T_map, p_values, weights


if __name__ == "__main__":
   print('This file is intented to be used as imported only')

