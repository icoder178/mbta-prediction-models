import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma,norm
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.optimize import curve_fit

delay_inputs=pd.read_csv('~/mbta-prediction-models/data/analysis_data/delay_inputs.csv')
gse_inputs=pd.read_csv('~/mbta-prediction-models/data/analysis_data/GSE_inputs.csv')


#Rice Rule- 2*n^(1/3) for bins
plt.hist(gse_inputs['Gated_Station_Entries'],bins=32,density=True)
ax=plt.gca()
ax.set_xlabel('Gated Station Entries')
ax.set_ylabel('Frequency')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Gated Station Entries (n=32)')

def bimodal_gaussian(x,mu1,std1,A1,mu2,std2,A2):
    return (A1 * norm.pdf(x,mu1,std1) +
            A2 *norm.pdf(x,mu2,std2))

# Get histogram bin edges and counts for fitting
counts, bin_edges = np.histogram(gse_inputs['Gated_Station_Entries'].to_numpy(), bins=32, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

params, _ = curve_fit(bimodal_gaussian, bin_centers, counts,p0=[gse_inputs['Gated_Station_Entries'].quantile(0.25), gse_inputs['Gated_Station_Entries'].std() / 2,
      0.05, gse_inputs['Gated_Station_Entries'].quantile(0.75), gse_inputs['Gated_Station_Entries'].std() / 2, 0.05])

x= np.linspace(min(gse_inputs['Gated_Station_Entries']), max(gse_inputs['Gated_Station_Entries']), 100000)
y=bimodal_gaussian(x,*params)
plt.plot(x,y)
plt.show()

#Rice Rule- 24 bins
plt.hist(delay_inputs['Total_Delays'],bins=24,density=True)
ax_=plt.gca()
ax_.set_xlabel('Total Delays Per Day')
ax_.set_ylabel('Frequency')
ax_.spines['top'].set_visible(False)
ax_.spines['right'].set_visible(False)
ax_.set_title('Daily Delay Aggregates (n=24)')


shape,loc,scale=gamma.fit(delay_inputs['Total_Delays'])
x=np.linspace(0,500,150)
y=gamma.pdf(x,a=shape,loc=loc,scale=scale)
print(shape,loc,scale)
plt.plot(x,y)
plt.show()





