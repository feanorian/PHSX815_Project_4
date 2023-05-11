import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap
import scipy.stats as st

if __name__ == "__main__":


	if '-h' in sys.argv or '--help' in sys.argv:
		print ("Usage: %s [-t -n]" % sys.argv[0])
		print
		sys.exit(1)
	if '-f' in sys.argv:
		p = sys.argv.index('-f')
		InputFile = sys.argv[p+1]
	else:
		InputFile = 'mbh.csv'
	if '-mass' in sys.argv:
		p = sys.argv.index('-mass')
		mass = float(sys.argv[p+1])
	else:
		mass = 7


	np.random.seed(534)
	rng = np.random.default_rng()
	
	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		return array[idx]
	
	#InputFile = 'mbh.csv'
	with open(InputFile) as file:
		df=pd.read_csv(file)

	# removes entries for which we have no mass data
	clean_df=df[df['log M_BH'] != 'NAN']

	# converts the mass column to numerical data
	clean_df['log M_BH'] = pd.to_numeric(clean_df['log M_BH'])
	clean_df['log Z'] = np.log(clean_df['Redshift'])

	# Cut of AGN data by mass

	small_bh = clean_df[clean_df['log M_BH'] <= mass]
	large_bh = clean_df[clean_df['log M_BH'] > mass]

	data1 = (large_bh['log Z'],)
	data2 = (small_bh['log Z'],)
	

	# bootstrap sample for the data
	small_boot_std_z = bootstrap(data1, np.std, confidence_level=0.95,random_state=rng)
	large_boot_std_z = bootstrap(data2, np.std, confidence_level=0.95,random_state=rng)
	small_boot_mean_z = bootstrap(data1, np.mean, confidence_level=0.95,random_state=rng)
	large_boot_mean_z = bootstrap(data2, np.mean, confidence_level=0.95,random_state=rng)

	# log(Z) and stdev for small and large mass populations computed from bootstrap
	small_boot_std_z_data = small_boot_std_z.bootstrap_distribution
	small_boot_mean_z_data = small_boot_mean_z.bootstrap_distribution
	large_boot_std_z_data = large_boot_std_z.bootstrap_distribution
	large_boot_mean_z_data = large_boot_mean_z.bootstrap_distribution

	small_mean_z_true = np.mean(small_boot_mean_z_data)
	small_sigma_true = np.mean(small_boot_std_z_data)
	large_mean_z_true = np.mean(large_boot_mean_z_data)
	large_sigma_true = np.mean(large_boot_std_z_data)
	# Range of redshifts
	x = np.linspace(-7, 1, 10000, endpoint=True)

	likelihood_H0 = [st.norm.pdf(data1,i,large_sigma_true).sum() for i in x]
	likelihood_H1 = [st.norm.pdf(data2,i,large_sigma_true).sum() for i in x]

	likelihood_H0 = np.asarray(likelihood_H0)
	likelihood_H1 = np.asarray(likelihood_H1)
	likelihoodr_H0 = np.log10(likelihood_H0/likelihood_H1)
	likelihoodr_H1 = np.log10(likelihood_H1/likelihood_H0)

	crit = list(likelihoodr_H0).index(find_nearest(likelihoodr_H0, value=0.0))
	reject = round(find_nearest(likelihoodr_H0, value=0.0), 4)
	beta = (likelihoodr_H0 == likelihoodr_H0[crit]).sum()  / len(likelihoodr_H0)
	# confidence interval
	S_95 = small_boot_mean_z.confidence_interval
	L_95 = large_boot_mean_z.confidence_interval


	textstr = '\n'.join((
    	rf'$\beta$ = {round(beta, 4)}',
    	rf'power = {round((1-beta), 4)}'))

	sns.histplot(likelihoodr_H0, stat='probability',bins=50, element="step",fill=True, alpha=.3 ,color = 'aqua' , label = rf'$P(log_{{M_{{BH}}}} \leq {mass} | H0$)')
	sns.histplot(likelihoodr_H1, stat='probability',bins=50, element="step",fill=True, alpha=.3,color = 'salmon', label = rf'$P(log_{{M_{{BH}}}} \leq {mass} | H1$)')
	plt.title('Log Likelihood Ratios for H0 and H1')
	plt.xlabel('log (L(H0)/L(H1))')
	plt.yscale('log')
	plt.axvline(x = find_nearest(likelihoodr_H0, value=0.0), color='r', label=rf'LH0 = LH1')
	plt.annotate(textstr, xy=(0.05, 0.65), xycoords='axes fraction')
	plt.legend()
	#plt.savefig('LLR')
	plt.show()
