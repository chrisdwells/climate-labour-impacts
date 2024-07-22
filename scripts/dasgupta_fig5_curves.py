import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats

exposures = ['low', 'high']
ptiles = ['min', 'mean', 'max']

impacts = {}
for exposure in exposures:
    impacts[exposure] = {}
    
impacts['low']['min'] = [-23.1, -32.6, -48.8]
impacts['low']['mean'] = [-6.7, -10.3, -18.3]
impacts['low']['max'] = [3.1, 3.6, 5.3]

impacts['high']['min'] = [-31.6, -44.8, -45.5]
impacts['high']['mean'] = [-10.1, -14.9, -24.8]
impacts['high']['max'] = [4.3, 4.7, 7.0]

# Not sure if we trust impacts['high']['min'] 3K -45.5; when fitting it stands
# out. Instead try predicting based on ratio of high/low exposure from 2K

high_min_3k = -48.8/(-32.6/-44.8)
impacts['high']['min'] = [-31.6, -44.8, high_min_3k]


temps_cf_pi = [1.5, 2.0, 3.0]

# use the 2022 indicators paper as used in FRIDA calibration
t_offset_1986_2005 = 0.688948529

temps_cf_1986_2005 = temps_cf_pi - np.asarray(t_offset_1986_2005)

temps_plot = np.linspace(0, 2.5, 100)

colors = {
    'low':'blue',
    'high':'red',
    }

linestyles = {
    'min':'dotted',
    'mean':'solid',
    'max':'dotted',
    }

def fit(x, beta_t, beta_t2):
    yhat = beta_t*x + beta_t2*x**2
    return yhat


params = {}
for exposure in exposures:
    params[exposure] = {}
    for ptile in ptiles:
        
        plt.scatter(temps_cf_pi, impacts[exposure][ptile], color=colors[exposure])

        
        params_in, _ = curve_fit(
            fit, temps_cf_1986_2005, impacts[exposure][ptile])
        
        params[exposure][ptile] = params_in
        
        plt.plot(temps_plot+t_offset_1986_2005, fit(temps_plot, *params_in), 
                 color=colors[exposure], linestyle=linestyles[ptile])
        

#%%

def opt(x, q05_desired, q50_desired, q95_desired):
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


dist_params = {}
for exposure in exposures:
    dist_params[exposure] = {}
    for t_i, t in enumerate(temps_cf_pi):
        
        q05_in = impacts[exposure]['min'][t_i]
        q50_in = impacts[exposure]['mean'][t_i]
        q95_in = impacts[exposure]['max'][t_i]
        
        params_in = scipy.optimize.root(opt, [1, 1, 1], 
                    args=(q05_in, q50_in, q95_in)).x
        
        dist_params[exposure][t_i] = params_in


#%%
percentiles = np.linspace(0.05, 0.95, 19)

percentiles = np.asarray([0.1, 0.5, 0.9])

linestyle_list = ['dotted', 'solid', 'dashed']

for perc_i, percentile in enumerate(percentiles):
    for exposure in exposures:
        vals = []
        for t_i, t in enumerate(temps_cf_pi):
            params_dist = dist_params[exposure][t_i]
            
            vals.append(scipy.stats.skewnorm.ppf(percentile, 
                         params_dist[0], params_dist[1], params_dist[2]))
            
            
        params_percentile, _ = curve_fit(
            fit, temps_cf_1986_2005, vals)
             

        plt.plot(temps_plot+t_offset_1986_2005, fit(temps_plot, *params_percentile), 
                 color=colors[exposure], linestyle = linestyle_list[perc_i],
                 label=f'{exposure} exposure, {100*percentile}')
      
plt.legend()

#%%


skew_test = scipy.stats.skewnorm.rvs(params_in[0], params_in[1], params_in[2], 
                              size=10**5)/q50_in

xs = np.linspace(-40, 10, 1000)

plt.plot(xs, scipy.stats.skewnorm.cdf(xs, params_in[0], params_in[1], params_in[2]))
plt.scatter([q05_in, q50_in, q95_in], [0.05, 0.5, 0.95])

scipy.stats.skewnorm.cdf([q05_in, q50_in, q95_in], params_in[0], params_in[1], params_in[2])

