import numpy
import MA_estimators
import importlib
importlib.reload(MA_estimators) # reupdate imported codes, useful for debugging

def get_MA_outputs(contrast_estimates):
	# Store results in:
	results_simulation = {}
	
	J = contrast_estimates.shape[1]

	def run_estimator(title, estimator_function):
		nonlocal results_simulation
		print(f"Running -{title}- estimator")
		T_map, p_values = estimator_function(contrast_estimates)
		ratio_significance_raw = (p_values <= 0.05).sum() / len(p_values)
		ratio_significance = numpy.round(ratio_significance_raw * 100, 4)
		lim = 2 * numpy.sqrt(0.05 * (1 - 0.05) / J)
		verdict = 0.05 - lim <= ratio_significance_raw <= 0.05 + lim
		results_simulation[title] = {
		   'T_map': T_map,
		   'p_values': p_values,
		   'ratio_significance': ratio_significance,
		   'verdict': verdict
		} 
	run_estimator("Average", MA_estimators.average)
	run_estimator("Stouffer", MA_estimators.Stouffer)
	run_estimator("Dependence-Corrected \nStouffer", MA_estimators.dependence_corrected_Stouffer)
	run_estimator("GLS Stouffer", MA_estimators.GLS_Stouffer)
	run_estimator("Consensus Stouffer", MA_estimators.consensus_Stouffer)
	run_estimator("Consensus Weighted \nStouffer", MA_estimators.weighted_Stouffer)
	run_estimator("Consensus GLS \nStouffer", MA_estimators.consensus_GLS_Stouffer)
	run_estimator("Consensus Average", MA_estimators.consensus_average)

	return results_simulation

if __name__ == "__main__":
   print('This file is intented to be used as imported only')





