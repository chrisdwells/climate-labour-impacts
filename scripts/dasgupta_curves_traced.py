import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.lines import Line2D

FIGDIR = '../plots'

exposures = ['low', 'high']
# ptiles = ['0', '25', '50', '75', '100']
ptiles = ['25', '50', '75']

impacts = {}
for exposure in exposures:
    impacts[exposure] = {}
    
# before, used data from their supp tables. but don't trust this - it 
# doesnt match their plots. so here, take the 0, 25, 50, 75, 100 %iles
# traced from their graphs, which we should be able to trust.
# one issue is that this seems to be the spread between gridpoints...
    
# impacts['low']['0'] = [2.95, 3.48, 5.21]
impacts['low']['25'] = [-1.97, -3.30, -6.36]
impacts['low']['50'] = [-6.50, -10.62, -20.33]
impacts['low']['75'] = [-10.22, -15.68, -27.65]
# impacts['low']['100'] = [-22.06, -32.71, -48.80]

# impacts['high']['0'] = [4.26, 4.73, 7.09]
impacts['high']['25'] = [-3.55, -5.67, -11.11]
impacts['high']['50'] = [-11.35, -16.78, -27.90]
impacts['high']['75'] = [-15.84, -22.70, -35.70]
# impacts['high']['100'] = [-31.69, -44.92, -66.19]


temps_cf_1986_2005 = [1.5, 2.0, 3.0]

# use the 2022 indicators paper as used in FRIDA calibration
t_offset_pi_to_1986_2005 = 0.688948529

temps_cf_pi = temps_cf_1986_2005 + np.asarray(t_offset_pi_to_1986_2005)

temps_plot = np.linspace(0, 4, 100)

colors = {
    'low':'blue',
    'high':'red',
    }

def fit1(x, alpha, beta_t, beta_t2):
    yhat = alpha + beta_t*x + beta_t2*x**2
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
for fit in [fit1]:#[fit1, fit2]:
    handles.append(Line2D([0], [0], label=fitname, color='grey'))

for exposure in exposures:
    handles.append(Line2D([0], [0], label=exposure, color=colors[exposure]))

plt.legend(handles=handles)
plt.xlabel('GMST cf 1986-2005')
plt.ylabel('Labour Response cf 1986-2005')

plt.tight_layout()
# plt.savefig(f'{FIGDIR}/dasgupta_traced.png', dpi=300)
# plt.clf()

#%%

def opt(x, q25_desired, q50_desired, q75_desired):
    q25, q50, q75 = scipy.stats.skewnorm.ppf(
        (0.25, 0.50, 0.75), x[0], loc=x[1], scale=x[2]
    )
    return (q25 - q25_desired, q50 - q50_desired, q75 - q75_desired)

temps_stats = np.linspace(0.5, 4.5, 9) 


dist_params = {}
for impact in impacts.keys():
    dist_params[impact] = {}
        
    for t in temps_stats:
        
        q25_in = fit2(t, *params[impact]['quadratic']['25'])
        q50_in = fit2(t, *params[impact]['quadratic']['50'])
        q75_in = fit2(t, *params[impact]['quadratic']['75'])
    
        params_in = scipy.optimize.root(opt, [1, 1, 1], 
                    args=(q25_in, q50_in, q75_in)).x
            
        dist_params[impact][t] = params_in

#%%

percentiles = np.linspace(0.05, 0.95, 19)

percentiles = np.asarray([0.25, 0.5, 0.75])

linestyle_list = ['dotted', 'solid', 'dashed']

for impact in impacts.keys():
    
    for perc_i, percentile in enumerate(percentiles):
        vals = []
        for t in temps_stats:
            params_dist = dist_params[impact][t]
            
            vals.append(scipy.stats.skewnorm.ppf(percentile, 
                         params_dist[0], params_dist[1], params_dist[2]))
            
            
        params_percentile, _ = curve_fit(
            fit2, temps_stats, vals)
             
    
        plt.plot(temps_plot, fit2(temps_plot, *params_percentile), 
                 linestyle = linestyle_list[perc_i],
                 label=f'{impact} {100*percentile}', color=colors[impact])
