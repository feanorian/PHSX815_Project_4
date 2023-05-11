# PHSX815_Project_4
This project uses data from the  <a href="https://iopscience.iop.org/article/10.1086/679601">AGN Black Hole Database</a> to do exploratory data analysis
on the relatioship between redshift and black hole mass for AGNs. The likelihood of an AGN being at a higher or lower redshift based on its mass is also determined. Data
files are also included in the repository to use the code. Eachg .csv file corresponds to a different determination of the black hole mass based on reverberation mapping.

http://www.astro.gsu.edu/AGNmass/
# agn_eda.py
Exploratory data analysis for AGN black hole masses
## Usage:

`python3 agn_eda.py -f filname -mass mass`

# ll_analysis.py

## Usage:

`python3 ll_analysis.py -f filname -mass mass`

`-f`: filename for the data

`-mass`: This will be the mass split of the data. These will be orders of magnitude. Practically for AGN's, this will be a float between 6.0 and 10.5

# NOTE: Ensure you have the latest version of scipy installed (version 1.10.0)
