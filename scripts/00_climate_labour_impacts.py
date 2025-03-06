import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# import scipy.stats
from matplotlib.lines import Line2D
import pandas as pd
import pickle

# We use Dasgupta et al. 2021
# https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(21)00170-4/fulltext
# Supplementary table data doesn't seem to match the plots, so we trace the 
# graphs. The uncertainty seems to be the spread between gridpoints - so we can't
# use this for uncertainties in the global response. So for now, we just 
# provide the median by default - though the data is here for the others.

figdir = '../figures'
datadir = '../data'

exposures = ['low', 'high']
# ptiles = ['0', '25', '50', '75', '100']
ptiles = ['50']

impacts = {}
for exposure in exposures:
    impacts[exposure] = {}
    exp_file = f'{datadir}/dasgupta_{exposure}.csv'
    exp_in = pd.read_csv(exp_file)
    for ptile in ptiles:
        impacts[exposure][ptile] = exp_in.loc[exp_in['Percentile'] == float(ptile)].values[0,1:]
    
temps_cf_pi = [1.5, 2.0, 3.0]

temps_plot = np.linspace(0, 4, 100)

colors = {
    'low':'blue',
    'high':'red',
    }

def fit1(x, beta_t, beta_t2):
    yhat = beta_t*x + beta_t2*x**2
    return yhat

fitname = 'Quadratic'


params = {}
for exposure in exposures:
    params[exposure] = {}
    params[exposure][fitname] = {}

    for ptile in ptiles:
        
        plt.scatter(temps_cf_pi, impacts[exposure][ptile], color=colors[exposure])

        
        params_in, _ = curve_fit(
            fit1, temps_cf_pi, impacts[exposure][ptile])
        
        params[exposure][fitname][ptile] = params_in
        
        plt.plot(temps_plot, fit1(temps_plot, *params_in), 
                 color=colors[exposure])
            
handles = []
# for fit in [fit1]:#[fit1, fit2]:
#     handles.append(Line2D([0], [0], label=fitname, color='grey'))

for exposure in exposures:
    handles.append(Line2D([0], [0], label=exposure, color=colors[exposure]))

plt.legend(handles=handles)
plt.xlabel('GMST cf pre-industrial')
plt.ylabel('Labour Response cf pre-industrial')

plt.tight_layout()
# plt.savefig(f'{figdir}/dasgupta_traced_median_only.png', dpi=300)
# plt.clf()

#%%


with open(f'{datadir}/outputs/params.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)



#%%

# def opt(x, q25_desired, q50_desired, q75_desired):
#     q25, q50, q75 = scipy.stats.skewnorm.ppf(
#         (0.25, 0.50, 0.75), x[0], loc=x[1], scale=x[2]
#     )
#     return (q25 - q25_desired, q50 - q50_desired, q75 - q75_desired)

# temps_stats = np.linspace(0.5, 4.5, 9) 


# dist_params = {}
# for impact in impacts.keys():
#     dist_params[impact] = {}
        
#     for t in temps_stats:
        
#         q25_in = fit2(t, *params[impact]['quadratic']['25'])
#         q50_in = fit2(t, *params[impact]['quadratic']['50'])
#         q75_in = fit2(t, *params[impact]['quadratic']['75'])
    
#         params_in = scipy.optimize.root(opt, [1, 1, 1], 
#                     args=(q25_in, q50_in, q75_in)).x
            
#         dist_params[impact][t] = params_in

# #%%

# percentiles = np.linspace(0.05, 0.95, 19)

# percentiles = np.asarray([0.25, 0.5, 0.75])

# linestyle_list = ['dotted', 'solid', 'dashed']

# for impact in impacts.keys():
    
#     for perc_i, percentile in enumerate(percentiles):
#         vals = []
#         for t in temps_stats:
#             params_dist = dist_params[impact][t]
            
#             vals.append(scipy.stats.skewnorm.ppf(percentile, 
#                          params_dist[0], params_dist[1], params_dist[2]))
            
            
#         params_percentile, _ = curve_fit(
#             fit2, temps_stats, vals)
             
    
#         plt.plot(temps_plot, fit2(temps_plot, *params_percentile), 
#                  linestyle = linestyle_list[perc_i],
#                  label=f'{impact} {100*percentile}', color=colors[impact])
