import numpy
import MA_estimators

####################################
##### COMPUTE META ANALYSIS ESTIMATORS 
##### FOR A GIVEN SET OF CONTRAST ESTIMATES
##### AND SEND RESULTS AS A DICT
##### Including T values, p values, 
##### ratio of significant p values,
##### the verdict (True or False) for
##### the simulations as well as the weights
##### for the study of SDMA GLS and SDMA Stouffer
####################################


def get_MA_outputs(contrast_estimates):
	# Store results in:
	results_simulation = {}
	J = contrast_estimates.shape[1]

	def run_estimator(title, estimator_function):
		nonlocal results_simulation
		print(f"Running -{title}- estimator")
		T_map, p_values, weights = estimator_function(contrast_estimates)
		ratio_significance_raw = (p_values <= 0.05).sum() / len(p_values)
		ratio_significance = numpy.round(ratio_significance_raw * 100, 4)
		lim = 2 * numpy.sqrt(0.05 * (1 - 0.05) / J)
		verdict = 0.05 - lim <= ratio_significance_raw <= 0.05 + lim
		results_simulation[title] = {
		   'T_map': T_map,
		   'p_values': p_values,
		   'ratio_significance': ratio_significance,
		   'verdict': verdict,
		   'weights': weights
		} 
	run_estimator("Stouffer", MA_estimators.Stouffer)
	run_estimator("SDMA Stouffer", MA_estimators.SDMA_Stouffer)
	run_estimator("Consensus \nSDMA Stouffer", MA_estimators.Consensus_SDMA_Stouffer)
	run_estimator("Consensus Average", MA_estimators.Consensus_Average)
	run_estimator("SDMA GLS", MA_estimators.SDMA_GLS)
	run_estimator("Consensus SDMA GLS", MA_estimators.Consensus_SDMA_GLS)
	

	return results_simulation

if __name__ == "__main__":
   print('This file is intented to be used as imported only')





