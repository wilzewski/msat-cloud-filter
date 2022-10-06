# -*- coding: utf-8 -*-
# Cloud filter for MethaneAIR/MethaneSAT
# Jonas Wilzewski (jwilzewski@g.harvard.edu)

class cloudFilter(object):

    def __init__(self, l1, msi, glint=False, osse=False):

        from netCDF4 import Dataset
        import numpy as np
        import os 
        import glob
        import sys

        '''
        Initialize msat-cloud-filter.
        ARGS:
            osse (bool): synthetic data (True) or 
                         MAIR/MSAT measurements (False)
            l1 (string): path to L1 data file or data directory
            msi (string): path to MSI data
            glint (bool): ocean glint?
            truth_path (string): path to OSSE truth data dir
        '''

        if osse:
            self.data_type=0
            # if OSSE always use geos-fp profile data as truth:
            self.truth_path = '/scratch/sao_atmos/ahsouri/Retrieval/methanesat_osses/profile_prep/geos-fp-profile/airglow_extended/'
            # check if level1 is file or directory:
            if os.path.isdir(os.path.abspath(l1)):
                self.l1_bundle = sorted(glob.glob(l1 + '/*.nc'))
                if len(self.l1_bundle)==0:
                    print('Provide L1 directory containing netCDF data files')
                    sys.exit()
            else:
                # l1 is one file
                self.l1_bundle = []
                self.l1_bundle.append(os.path.abspath(l1))

            # Initialize
            filter_shape = (Dataset(self.l1_bundle[0]).dimensions['jmx'].size,
                             Dataset(self.l1_bundle[0]).dimensions['imx'].size)
            self.prefilter = np.zeros(filter_shape)
            self.quality_flag = np.zeros(filter_shape)
            self.postfilter = np.zeros(filter_shape)
            self.l1_ref = np.zeros(filter_shape)
            self.l1_sza = np.zeros(filter_shape)
            self.l1_alb = np.zeros(filter_shape)
            self.msi_ref = np.zeros(filter_shape)
            self.cloud_truth = np.zeros(filter_shape)
            self.msi_path = msi
            
        else:
            # MSAT/MAIR data
            self.data_type = 1
            # TODO fill in initialization for MAIR data
            
    def get_prefilter(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import sys

        print('read_l1()')
        self.l1_sza, self.l1_alb, self.l1_ref, \
             self.cloud_truth, l1_lon, l1_lat = self.read_l1()

        print('read_msi()')
        self.msi_ref = self.read_msi(l1_lon, l1_lat)
        
    def read_msi(self, l1_lon, l1_lat):
        import numpy as np
        import rasterio
        from scipy.interpolate import griddata
        import os
        import pickle
        import sys
        import matplotlib.pyplot as plt

        if self.data_type==0:
            # This function collects MSI data on a grid
            # and interpolates an average reflectance to each
            # l1 pixel using a fixed pixel size (as a first approximation
            # since l1 pixels in OSSE are unevenly spaced). 

            l1_pseudo_pixel_size = 0.3   # somewhat representative value for OSSE
            grid_spacing = l1_pseudo_pixel_size/2    # what is the best choice here?

            save_file = 'msi_clim_0res15.pkl'
            if save_file not in os.listdir():
                print('Generating MSI Climatology')

                #l1_lon_min = np.nanmin(l1_lon[np.abs(l1_lon)<=180])
                #l1_lon_max = np.nanmax(l1_lon[np.abs(l1_lon)<=180])
                #l1_lat_min = np.nanmin(l1_lat[np.abs(l1_lat)<=90])
                #l1_lat_max = np.nanmax(l1_lat[np.abs(l1_lat)<=90])

                # set up grid on which to collect msi climatology data
                #b = 2 * l1_pseudo_pixel_size # use a buffer for edge pixels
                #lon = np.arange(l1_lon_min - b, l1_lon_max + b, grid_spacing)
                #lat = np.arange(l1_lat_min - b, l1_lat_max + b, grid_spacing)
                lon_min, lon_max, lat_min, lat_max = -180.0, 180.0, -90.0, 90.0
                lon = np.arange(lon_min, lon_max, grid_spacing)
                lat = np.arange(lat_min, lat_max, grid_spacing)
                lon,lat = np.meshgrid(lon, lat)
                
                msi_clim = np.zeros(lon.shape)
                c=0
                for f in sorted(os.listdir(self.msi_path)):
                    data = rasterio.open(self.msi_path+'/'+f, crs='EPSG:3857')
                    img = data.read(1)
                    if len(np.where(np.isnan(img)==False)[0])==0:
                        continue
                    else:
                        width = data.width
                        height = data.height
                        trn = data.transform
                        corner1 = trn * (0,0)
                        corner4 = trn * (height,width)
                        min_lat = corner4[1]
                        max_lat = corner1[1]
                        min_lon = corner1[0]
                        max_lon = corner4[0]
                        # skip file it is not fully inside l1 area
                        if min_lon<lon_min or max_lon>lon_max or\
                           min_lat<lat_min or max_lat>lat_max:
                            continue
                        c+=1
                        #dlon = (corner4[0]-corner1[0])/width
                        #dlat = (corner1[1]-corner4[1])/height
                        msi_lon, msi_lat = np.meshgrid(np.linspace(min_lon, max_lon, width),\
                                                       np.linspace(min_lat, max_lat, height))
                        #print(lon, lat)
                        #print('\n', msi_lon.shape, msi_lat.shape, img.shape);sys.exit()
                        msi = griddata((msi_lon.ravel(), msi_lat.ravel()), img.ravel(), (lon, lat))
                        msi[np.isnan(msi)] = 0.0
                        msi[msi_clim!=0] = 0.0          # no double registration
                        msi_clim = msi_clim + msi
                        #plt.pcolor(lon, lat, msi_clim);plt.show()
                        if c>1:
                            break
                pkl_file = open(save_file, 'wb')
                pickle.dump([lon, lat, msi_clim], pkl_file)
                pkl_file.close()
            else:
                pkl_file = open(save_file, 'rb')
                data = pickle.load(pkl_file)
                lon = data[0]; lat = data[1]; msi_clim = data[2]
                pkl_file.close()

                l1_lon[np.abs(l1_lon)>180] = np.nan
                l1_lat[np.abs(l1_lat)>90] = np.nan

                #poly = make_polygons(l1_lon, l1_lat)
                
                plt.pcolor(lon, lat, msi_clim)
                plt.scatter(l1_lon, l1_lat, c=self.l1_sza, s=1)
                plt.show()

                msi_ref = np.zeros(l1_lon.shape)
                
                #msi_ref = griddata((lon.ravel(), lat.ravel()), msi_clim.ravel(), (l1_lon, l1_lat))
                msi_ref[np.isnan(l1_lon)]=np.nan
                #print(msi_ref)
                #print(msi_ref.shape)
                    
                    

        else:
            msi_ref = np.zeros(self.prefilter.shape)
            # TODO implement msi reader for measurement data

        return msi_ref

       
    def read_l1(self):
        from netCDF4 import Dataset
        import sys
        import numpy as np

        sza, alb, lon, lat, xch4_true, cf, l1_ref, = [],[],[],[],[],[],[]

        if self.data_type==0:
            for f in self.l1_bundle:
                d = Dataset(f)
                ff = f.split('/')[-1].split('_')
                ix0 = int(ff[1])-1
                ixf = int(ff[2])
                it0 = int(ff[4])-1
                itf = int(ff[5].split('.')[0])

                sza_tmp = d.groups['Level1'].variables['SolarZenithAngle'][0,it0:itf,ix0:ixf].T.squeeze()
                # insert nan for missing pixels so that shape is consistent with input data shape
                sza_tmp = np.pad(sza_tmp, (it0,0), 'constant', constant_values=(np.nan))
                sza.append( sza_tmp )
                rad_tmp = d.groups['RTM_Band2'].variables['Radiance_I'][:,it0:itf,ix0:ixf].T
                rad_tmp = rad_tmp * d.groups['RTM_Band2'].variables['Irradiance'][:,it0:itf,ix0:ixf].T
                rad_tmp = rad_tmp.squeeze()
                rad_tmp = np.pad(rad_tmp, [(it0,0),(0,0)], 'constant', constant_values=(np.nan))

                # calculate reflectance spectra
                I0_CH4 = 1.963e14  # approx I0 @ 1621 nm [photons/s/cm2/nm] 
                                   # (from chance_jpl_mrg_hitran_200-16665nm.nc)
                l1_ref.append( (np.pi*np.array(rad_tmp).T / 
                              (np.cos(np.pi*np.array(sza_tmp)/180.)*I0_CH4)).T )

                alb_tmp = d.groups['OptProp_Band1'].variables['BRDF_KernelAmplitude_isotr'][:,it0:itf,ix0:ixf].T
                alb_tmp = np.mean(alb_tmp,axis=2).T.squeeze()   # why average?
                alb.append( np.pad(alb_tmp, (it0,0), 'constant', constant_values=(np.nan))  )

                lat_tmp = d.groups['Level1'].variables['Latitude'][0,it0:itf,ix0:ixf].T.squeeze()
                lat_tmp = np.pad(lat_tmp, (it0,0), 'constant', constant_values=(np.nan))
                lat.append( lat_tmp )
                lon_tmp= d.groups['Level1'].variables['Longitude'][0,it0:itf,ix0:ixf].T.squeeze()
                lon_tmp = np.pad(lon_tmp, (it0,0), 'constant', constant_values=(np.nan))
                lon.append( lon_tmp )

                #xch4_tmp = d.groups['Profile'].variables['CH4_ProxyMixingRatio'][0,it0:itf,ix0:ixf].T
                #xch4_tmp = xch4_tmp[:].T.squeeze()
                #xch4_true.append( np.pad(xch4_tmp, (it0,0), 'constant', constant_values=(np.nan)) )

                cf.append(self.get_cloud_fraction(self.truth_path, f, ix0, ixf))

            return np.array(sza).T, np.array(alb).T, np.array(l1_ref).T,\
                     np.array(cf).T, np.array(lon).T, np.array(lat).T #, np.array(xch4_true).T

        #else:   TODO write MAIR L1 reader
            

    def get_cloud_fraction(self, osse_path, l1_file, ix0, ixf):
        import os
        from netCDF4 import Dataset
        import numpy as np
        import sys

        cloud_fraction = []
        for f in os.listdir(osse_path):
            if l1_file.split('h_')[0].split('SAT')[1] in f:
                profile_input = Dataset(os.path.join(osse_path,f))
                cf = profile_input['SupportingData/CloudFraction'][:,ix0:ixf].squeeze()
                cloud_fraction.append(cf)
        if len(cloud_fraction[0]) != self.prefilter.shape[0]:
            print('OSSE Truth data wrong dimension');sys.exit()

        return np.array(cloud_fraction).squeeze()


