import numpy

def generate_simulation(case="Null", K=20, J=20000, corr=0.8, mean=2):
    numpy.random.seed(0) # reproducibility
    # simulation 0
    sigma=1
    # simulation 0 and 1
    mu = 0

    # Simulation 0: The dumbest, null case: independent pipelines, mean 0, variance 1 (totally iid data)
    # generate iid matrix of dimension K columns, J rows
    if case=="Null":
        print("RUNNING SIMULATION: ", case)
        rng = numpy.random.default_rng(seed=0)
        return mu + sigma * rng.standard_normal(size=(K,J))

    # Simulation 1: Null data with correlation: Induce correlation Q, mean 0, variance 1
    # => verifies that the 1’Q1/K^2 term is correctly accounting for dependence.
    elif case=="Null correlated":
        print("RUNNING SIMULATION: ", case)
        cov_mat = numpy.ones((K, K)) * corr
        numpy.fill_diagonal(cov_mat, 1)
        offset = numpy.zeros(K) # mean 0
        return numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J

    # Simulation 2: Non-Null data with correlation: Induce correlation Q, mean >= 1, variance 1
    elif case=="Non-null correlated":
        print("RUNNING SIMULATION: ", case)
        cov_mat = numpy.ones((K, K)) * corr
        numpy.fill_diagonal(cov_mat, 1)
        offset = numpy.ones(K) * mean 
        return numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J

    # Simulation 2: Non-null but totally heterogeneous data per team: Correlation Q + 
    # all pipelines share same mean, but 50% of voxels have mu=mu1, 50% of voxels have mu = -mu1
    elif case=="Non-null heterogeneous":
        print("RUNNING SIMULATION: ", case)
        cov_mat = numpy.ones((K, K)) * corr
        numpy.fill_diagonal(cov_mat, 1)
        offset = numpy.ones(K) * mean 
        matrix_data = numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J
        random_sign = numpy.random.choice([1, -1], size=J, replace=True)
        return matrix_data*random_sign

if __name__ == "__main__":
   print('This file is intented to be used as imported only')