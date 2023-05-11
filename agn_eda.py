"""
Name: Craig Brooks
PHSX 815 Spring 2023
Project 4
Due Date 5/8/2023
This code performs exploratory data analysis on AGN data from the AGN Black Hole Mass database
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
sys.path.append(".")
import scipy.stats as st
from scipy.stats import bootstrap

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
		mass = sys.argv[p+1]
	else:
		mass = '7'

	np.random.seed(534)
	rng = np.random.default_rng()

	InputFile = 'mbh.csv'
	with open(InputFile) as file:
		df=pd.read_csv(file)

	# removes entries for which we have no mass data
	clean_df=df[df['log M_BH'] != 'NAN']

	# converts the mass column to numerical data
	clean_df['log M_BH'] = pd.to_numeric(clean_df['log M_BH'])
	clean_df['log Z'] = np.log(clean_df['Redshift'])

	# Cut of AGN data by mass
	small_bh = clean_df[clean_df['log M_BH'] < float(mass)]
	large_bh = clean_df[clean_df['log M_BH'] >= float(mass)]

	# clean redshift data
	data1 = (small_bh['log Z'],)
	data2 = (large_bh['log Z'],)
	
	# Bootstrap om mean of log(Z) for small and large mass AGN
	small_boot_mean_z = bootstrap(data1, np.mean, confidence_level=0.95,random_state=rng)
	large_boot_mean_z = bootstrap(data2, np.mean, confidence_level=0.95,random_state=rng)

	# True Mean

	true_small = np.median(small_boot_mean_z.bootstrap_distribution)
	true_large = np.median(large_boot_mean_z.bootstrap_distribution)
	# STDEV of log (Z)
	small_boot_std_z = small_boot_mean_z.standard_error
	large_boot_std_z = large_boot_mean_z.standard_error

	# Distribution of log (Z)
	small_boot_mean_z_data = small_boot_mean_z.bootstrap_distribution
	large_boot_mean_z_data = large_boot_mean_z.bootstrap_distribution

	textstr = '\n'.join((
    	rf'$log{{Z}}_s:  {round(true_small, 4)} ,\sigma_{{z,s}}$ = {round(small_boot_std_z, 4)}',
    	rf'$log{{Z}}_l:  {round(true_large, 4)} ,\sigma_{{z,l}}$= {round(large_boot_std_z, 4)}'))

	# plot regression lines for total data
	ax = plt.axes()
	sns.regplot(x = 'log Z',y = 'log M_BH',data=clean_df, line_kws = {'color':'red'})
	plt.title('AGN log(M_BH) vs log(z)')
	plt.xlabel('Redshift (log(z))')
	#plt.savefig('total_reg')
	plt.show()

	
	# Histplot for mean of log(Z)
	fig1, ax1 = plt.subplots()
	sns.histplot(small_boot_mean_z_data, stat='probability',bins=25, color = 'salmon', element="step",fill=True, alpha=.3, label=rf'$M_{{BH}} < 10^{{{mass}}} M_{{Sun}}$')
	sns.histplot(large_boot_mean_z_data, bins=25, stat = 'probability',color = 'aqua', element="step",fill=True, alpha=.3, label=rf'$M_{{BH}} > 10^{{{mass}}} M_{{Sun}}$')
	ax1.set_title('Bootstrap Distribution for log(Z) based on AGN M_BH')
	ax1.set_xlabel('mean log (Z)')
	ax1.set_ylabel('probability')
	plt.annotate(textstr, xy=(0.05, 0.65), xycoords='axes fraction')
	plt.legend()
	#plt.savefig('hist_logz_boot')
	plt.show()
