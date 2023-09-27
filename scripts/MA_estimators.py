import numpy
import scipy
from sklearn.preprocessing import StandardScaler
import importlib
import utils

importlib.reload(utils) # reupdate imported codes, useful for debugging


####################################
##### META ANALYSIS ESTIMATORS #####
####################################

#########
##### Conventional fixed effects MA models #####
#########

def average(contrast_estimates):
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
    # final stats is sqrt(k*1/k*sum(Zi))
    T_map = numpy.sqrt(K)*1/K*numpy.sum(contrast_estimates, 0) # team wise
    T_map = T_map.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights


#########
##### SDMA fixed effects MA models #####
#########

def dependence_corrected_Stouffer(contrast_estimates):
    K = contrast_estimates.shape[0]
    ones = numpy.ones((K, 1))
    # Q1 = numpy.cov(contrast_estimates)
    Q0 = numpy.corrcoef(contrast_estimates)
    attenuated_variance = ones.T.dot(Q0).dot(ones) / K**2
    # compute meta-analytic statistics
    T_map = numpy.mean(contrast_estimates, 0)/numpy.sqrt(attenuated_variance)
    T_map = T_map.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights

def GLS_Stouffer(contrast_estimates):
    K = contrast_estimates.shape[0]
    Q0 = numpy.corrcoef(contrast_estimates)
    Q = Q0.copy()
    top = numpy.ones(K).dot(numpy.linalg.inv(Q)).dot(contrast_estimates)
    down = numpy.ones(K).dot(numpy.linalg.inv(Q)).dot(numpy.ones(K))
    T_map = top/numpy.sqrt(down)
    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)

    weights = numpy.ones(K).dot(numpy.linalg.inv(Q))
    return T_map, p_values, weights

#########
##### SDMA consensus MA models #####
#########

def consensus_Stouffer(contrast_estimates):
    K = contrast_estimates.shape[0]
    consensus_mean = numpy.mean(contrast_estimates, 1).sum() / K # scalar
    Q0 = numpy.corrcoef(contrast_estimates)
    attenuated_variance = numpy.ones(K).T.dot(Q0).dot(numpy.ones(K)) / K**2 # scaler
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


def weighted_Stouffer(contrast_estimates):
    K = contrast_estimates.shape[0]
    # variance of each pipeline is assumed to be equal but allowed to vary over space
    # compute a standardized map Z∗ k for each pipeline k
    scaler = StandardScaler()
    contrast_estimates_std_Kwise = scaler.fit_transform(contrast_estimates.T).T # scaling team wise and back to normal shape
    # z* = (z - z_mean) / s 
    # with s = image-wise var for pipeline k
    # with z_mean = image-wise mean for pipeline k
    # numpy.divide(numpy.subtract(contrast_estimates.T, contrast_estimates.mean(axis=1)), contrast_estimates.std(axis=1))
    # These standardized maps are averaged over pipelines to create a map Z∗
    Z_star_mean_j = numpy.mean(contrast_estimates_std_Kwise, 0) # shape J
    # image-wise mean of this average of standard maps is zero
    assert numpy.mean(Z_star_mean_j) < 0.0001
    # Q1 = numpy.cov(contrast_estimates)
    Q0 = numpy.corrcoef(contrast_estimates)
    # and which is finally standardized, scaled and shifted 
    # to the consensus standard deviation and mean
    consensus_var = numpy.var(contrast_estimates, 1).sum() / K # scalar 
    consensus_std = numpy.sqrt(consensus_var) # scalar
    consensus_mean = numpy.mean(contrast_estimates, 1).sum() / K # scalar
    attenuated_variance = numpy.ones(K).T.dot(Q0).dot(numpy.ones(K)) / K**2 # scaler
    Z_star_consensus = (Z_star_mean_j.reshape(-1, 1)/ numpy.sqrt(attenuated_variance).reshape(1, -1)) * consensus_std + consensus_mean
    T_map = Z_star_consensus.reshape(-1)
    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    
    weights = contrast_estimates.mean(axis=1)*-1 # higher == less weight
    return T_map, p_values, weights

def consensus_GLS_Stouffer(contrast_estimates):
    # compute GLS Stouffer first
    K = contrast_estimates.shape[0]
    Q0 = numpy.corrcoef(contrast_estimates)
    Q = Q0.copy()
    GLS_Stouffer_mean = numpy.ones(K).dot(numpy.linalg.inv(Q)).dot(contrast_estimates)

    # then compute the consensus GLS Stouffer
    consensus_mean = numpy.mean(contrast_estimates, 1).sum() / K # scalar
    top = GLS_Stouffer_mean - consensus_mean
    down = numpy.ones(K).dot(numpy.linalg.inv(Q)).dot(numpy.ones(K))
    consensus_GLS_Stouffer = top/numpy.sqrt(down) + consensus_mean
    T_map = consensus_GLS_Stouffer.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    
    weights = numpy.ones(K).dot(numpy.linalg.inv(Q))
    return T_map, p_values, weights


def consensus_average(contrast_estimates):
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

if __name__ == "__main__":
   print('This file is intented to be used as imported only')

