from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import cartopy.crs as ccrs
import matplotlib.cm

DATADIR = '../Data'
FIGDIR = '../plots'
scenarios = ['RCP26', 'RCP85']
work_intensities = ['low', 'moderate', 'high']


df_in = pd.read_csv(os.path.join(DATADIR, 'fig6a_data.csv'))
df_in_rcp26_low = pd.read_csv(os.path.join(DATADIR, 'fig6a_rcp26_low.csv'))

lons = np.arange(-179.75, 180.25, 0.5)
lats = np.arange(-89.75, 90.25, 0.5)
#%%
maps = {}
for scenario in scenarios:
    for work_intensity in work_intensities:
        df_crop = df_in.loc[(df_in['RCP'] == scenario) & 
                    (df_in['workload'] == f'{work_intensity} work intensity')]
        
        if scenario == 'RCP26':
            if work_intensity == 'low':
                df_crop = df_in_rcp26_low.loc[(df_in_rcp26_low['RCP'] == scenario) & 
                    (df_in_rcp26_low['workload'] == f'{work_intensity} work intensity')]
                
        array = np.full((360, 720), np.nan) 
        
        n_data = 0
        for index, row in df_crop.iterrows():
            idx_lon = np.where(lons == row['lon'])[0][0]
            idx_lat = np.where(lats == row['lat'])[0][0]
            
            if str(row['value']) != 'nan':
                array[idx_lat, idx_lon] = row['value']
                n_data += 1
            
        maps[f'{scenario}_{work_intensity}'] = array
        
        print(f'{scenario}_{work_intensity}: {n_data} from {len(df_crop)} rows')

 #%%

cmap = matplotlib.cm.magma
cmap.set_bad('lightgrey',1)

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.ravel()
for i in np.arange(6):
    axs[i].remove()

ax_count = 0
for scenario in scenarios:
    for work_intensity in work_intensities:
        ax_count += 1
    
        ax = plt.subplot(2, 3, ax_count, projection=ccrs.Robinson(
            central_longitude=0))

        mesh_1 = ax.pcolormesh(
            lons,
            lats,
            maps[f'{scenario}_{work_intensity}'],
            cmap=cmap,
            vmin=-40,
            vmax=0,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        
        cb = plt.colorbar(
            mesh_1, extend="neither", orientation="horizontal", shrink=0.6,
            pad=0.05, label = '% productivity change'
        )
        ax.coastlines()
        ax.set_title(f'{scenario}_{work_intensity}')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig6a_data.png', dpi=300)
# plt.clf()

#%%


nc_out = Dataset(os.path.join(DATADIR, 'fig6a_data.nc'), 'w')
latdim = nc_out.createDimension('latitude', 360)
londim = nc_out.createDimension('longitude', 720)
latvar = nc_out.createVariable('latitude', 'f4', ('latitude',))
lonvar = nc_out.createVariable('longitude', 'f4', ('longitude',))
latvar[:] = lats
lonvar[:] = lons 

for var in maps.keys():
    outvar = nc_out.createVariable(var, 'f4', ('latitude', 'longitude'))
    outvar[:] = maps[var]
    
latvar.axis = "Y"
latvar.units = "degrees_north"
latvar.standard_name = "latitude"
latvar.long_name = "latitude"
lonvar.axis = "X"
lonvar.units = "degrees_east"
lonvar.standard_name = "longitude"
lonvar.long_name = "longitude"

nc_out.close()

#%%

nc = Dataset(os.path.join(DATADIR, 'fig6a_data.nc'))

lats_nc = nc.variables['latitude'][:]
lons_nc = nc.variables['longitude'][:]

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.ravel()
for i in np.arange(6):
    axs[i].remove()

ax_count = 0
for scenario in scenarios:
    for work_intensity in work_intensities:
        ax_count += 1
    
        ax = plt.subplot(2, 3, ax_count, projection=ccrs.Robinson(
            central_longitude=0))

        mesh_1 = ax.pcolormesh(
            lons,
            lats,
            nc.variables[f'{scenario}_{work_intensity}'][:],
            cmap=cmap,
            vmin=-40,
            vmax=0,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        
        cb = plt.colorbar(
            mesh_1, extend="neither", orientation="horizontal", shrink=0.6,
            pad=0.05, label = '% productivity change'
        )
        ax.coastlines()
        ax.set_title(f'{scenario}_{work_intensity}')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig6a_data_nc.png', dpi=300)
# plt.clf()


