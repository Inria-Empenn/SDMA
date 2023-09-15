import numpy

def generate_simulation(case="Null", K=20, J=20000, corr=0.8, mean=2, anticorrelated_result_ratio=0.3, seed=0):
    numpy.random.seed(seed) # reproducibility
    # simulation 0
    sigma=1
    # simulation 0 and 1
    mu = 0

    # Simulation 0: The dumbest, null case: independent pipelines, mean 0, variance 1 (totally iid data)
    # generate iid matrix of dimension K columns, J rows
    if case=="Null":
        print("Generating simulation: ", case)
        rng = numpy.random.default_rng(seed=0)
        return mu + sigma * rng.standard_normal(size=(K,J))

    # Simulation 1: Null data with correlation: Induce correlation Q, mean 0, variance 1
    # => verifies that the 1’Q1/K^2 term is correctly accounting for dependence.
    elif case=="Null correlated":
        print("Generating simulation: ", case, ", corr=", corr)
        cov_mat = numpy.ones((K, K)) * corr
        numpy.fill_diagonal(cov_mat, 1)
        offset = numpy.zeros(K) # mean 0
        return numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J

    # Simulation 2: Non-Null data with correlation: Induce correlation Q, mean >= 1, variance 1
    elif case=="Non-null correlated":
        print("Generating simulation: ", case, ", corr=", corr)
        cov_mat = numpy.ones((K, K)) * corr
        numpy.fill_diagonal(cov_mat, 1)
        offset = numpy.ones(K) * mean 
        return numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J

    # Simulation 3: Non-null but heterogeneous data voxel-wise: Correlation Q + 
    # all pipelines share same mean, but 50% of voxels have mu=mu1, 50% of voxels have mu = -mu1
    elif case=="Non-null heterogeneous voxels":
        print("Generating simulation: ", case, ", corr=", corr)
        cov_mat = numpy.ones((K, K)) * corr
        numpy.fill_diagonal(cov_mat, 1)
        offset = numpy.ones(K) * mean 
        matrix_data = numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J
        random_sign = numpy.random.choice([1, -1], size=J, replace=True)
        return matrix_data*random_sign

    # Simulation 4: Non-null but heterogeneous data per team: Correlation Q + 
    # all pipelines share same mean, but a percentage (anticorrelated_result_ratio) of the team have mu = -mu1
    elif case=="Non-null heterogeneous pipelines":
        print("Generating simulation: ", case, ", corr=", corr, ", reverse ratio=", anticorrelated_result_ratio)
        cov_mat = numpy.ones((K, K)) * corr
        numpy.fill_diagonal(cov_mat, 1)
        offset = numpy.ones(K) * mean 
        matrix_data = numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J
        random_sign = numpy.random.choice([1, -1], size=K, replace=True, p=(1-anticorrelated_result_ratio, anticorrelated_result_ratio))
        matrix_data = matrix_data.T*random_sign # team wise heterogeneity
        return matrix_data.T # back to shape K, J


if __name__ == "__main__":
   print('This file is intented to be used as imported only')