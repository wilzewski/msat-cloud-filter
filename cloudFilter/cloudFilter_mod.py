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
            self.data_type = 0
            # if OSSE always use geos-fp profile data as truth:
            self.truth_path = '/scratch/sao_atmos/ahsouri/Retrieval/methanesat_osses/profile_prep/geos-fp-profile/airglow_extended/'
        else:
            # MSAT/MAIR data
            self.data_type = 1

        # check if level1 is file or directory:
        if os.path.isdir(os.path.abspath(l1)):
            self.l1_bundle = sorted(glob.glob(l1 + '/*.nc'))
            self.l1_name = l1.split('.')[1].split('.')[0]
            if len(self.l1_bundle)==0:
                print('Provide L1 directory containing netCDF data files')
                sys.exit()
        else:
            # l1 is one file
            self.l1_bundle = []
            self.l1_bundle.append(os.path.abspath(l1))
            self.l1_name = l1.split('/')[-2]

        # Initialize
        if osse:
            filter_shape = (Dataset(self.l1_bundle[0]).dimensions['jmx'].size,
                         Dataset(self.l1_bundle[0]).dimensions['imx'].size)
            self.cloud_truth = np.zeros(filter_shape)
        else:
            filter_shape = (Dataset(self.l1_bundle[0]).dimensions['x'].size,
                         Dataset(self.l1_bundle[0]).dimensions['y'].size)

        self.prefilter = np.zeros(filter_shape)
        self.quality_flag = np.zeros(filter_shape)
        self.postfilter = np.zeros(filter_shape)
        self.l1_ref = np.zeros(filter_shape)
        self.l1_sza = np.zeros(filter_shape)
        self.l1_alb = np.zeros(filter_shape)
        self.msi_ref = np.zeros(filter_shape)
        self.msi_path = msi
        self.prefilter_model = 0

    def apply_prefilter(self):

        print('read_l1()')
        self.l1_sza, self.l1_ref, l1_lon, l1_lat = self.read_l1()

        return self.prefilter


    def get_prefilter(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import sys
        import os
        import pickle

        if self.data_type==0:
            print('read_l1()')
            self.l1_sza, self.l1_alb, self.l1_ref, \
                 self.cloud_truth, l1_lon, l1_lat, l1_clon, l1_clat = self.read_l1()

            print('read_msi()')
            self.msi_ref = self.read_msi(l1_lon, l1_lat, l1_clon, l1_clat)

            # if model not present:
            if 'prefilter_model.pkl' not in os.listdir():
                print('build_prefilter_model()')
                self.prefilter_model = self.build_prefilter_model()
            else:
                print('Read prefilter model from prefilter_model.pkl')
                pkl_file = open('prefilter_model.pkl', 'rb')
                self.prefilter_model = pickle.load(pkl_file)
                pkl_file.close()
        else:
            if 'prefilter_model.pkl' not in os.listdir():
                print('Prefilter model not found. Create first (run code on OSSE).')
                sys.exit()
            else:
                print('Read prefilter model from prefilter_model.pkl')
                pkl_file = open('prefilter_model.pkl', 'rb')
                self.prefilter_model = pickle.load(pkl_file)
                pkl_file.close()

    def build_prefilter_model(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score
        import numpy as np
        import pickle

        X, y = self.prepare_prefilter_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # feature scaling so variables have mean=0 and std=1
        scaler = StandardScaler()
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # apply same transformation to test data
        X_test = scaler.transform(X_test)

        model = MLPClassifier(solver='lbfgs', max_iter=600, alpha=1e-5,
            hidden_layer_sizes=(10,), random_state=1)  # 10 neurons, 1 hidden layer

        # Train model
        model.fit(X_train, y_train)
        # Predict
        pred = model.predict(X_test)
        print('Accuracy: ', np.round(accuracy_score(y_test, pred),2))

        # save model
        save_file = 'prefilter_model.pkl'
        print('Saving prefilter model to ', save_file)
        pkl_file = open(save_file, 'wb')
        pickle.dump(model, pkl_file)
        pkl_file.close()

        return model

    def prepare_prefilter_data(self):
        import numpy as np
        import sys
        import matplotlib.pyplot as plt

        # clean up conditions: high sun, good spectra, no water
        good_data = np.where((self.l1_sza <= 70) &\
                             (np.nanmean(self.l1_ref, axis=0) <= 1)&\
                             (self.msi_ref > 0.001))

        msi = self.msi_ref[good_data]
        sza = self.l1_sza[good_data]
        ref = self.l1_ref[:, good_data[0], good_data[1]]
        cf = self.cloud_truth[good_data]

        if np.where(np.isnan(ref))[0].shape[0] >0:
            print('bad reflectance input'); sys.exit()

        # spectral sorting order
        order = self.spectral_sorting(ref, cf)
        ref = ref[order, :].T

        X = np.stack([sza, msi], axis=1)
        # append full, sorted reflectance spectra
        # X = np.hstack((X, ref))

        # alternatively, append average sorted spectra:
        nchannels = 5
        r=np.zeros((ref.shape[0],nchannels))
        l = int(ref.shape[1]/nchannels)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r[i,j] = np.mean(ref[i,j*l:(j+1)*l])
        X = np.hstack((X, r))

        # make binary label
        thresh = 0.9
        cf[cf>thresh] = 1
        cf[cf<=thresh] = 0

        return X, cf

    def spectral_sorting(self, ref, cf):
        # find sorting order
        import numpy as np
        import matplotlib.pyplot as plt

        ref_0 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]==0.0]), axis=0)
        order = np.argsort(ref_0)
        #plt.title('N='+str(ref.shape[1]))
        #plt.plot(np.linspace(0,1,ref_0.shape[0]), ref_0)
        #plt.plot(np.linspace(0,1,ref_0.shape[0]), ref_0[order]);plt.show()
        return order

    def av_msi_per_l1(self, l1_lon, msi_clim, points, c11, c12, c21, c22, c31, c32, c41, c42):
        # make mask indicating which MSI points lie within a l1 pixel
        # with corners c1-4; reshape mask given out_shape and
        # average MSI data over the l1 pixels
        # ARGS:
        #    l1_lon: longitude of L1 pixel center (float)
        #    msi_clim: climatology of MSI on some grid (2d array)
        #    points: center points of gridded MSI data (shape (X,2), with each row containing float(lon), float(lat))
        #    ci0: corner i longitude (L1)
        #    ci1: corner i latitude (L1)
        from matplotlib.path import Path
        import numpy as np

        if np.isnan(l1_lon):
            # if l1_lon is nan add 0 reflectance
            return 0
        else:
            # L1 pixel corner coordinates
            c1 = (c11, c12)
            c2 = (c21, c22)
            c3 = (c31, c32)
            c4 = (c41, c42)
            p = Path([c1, c2, c3, c4, c1])
            # find MSI pixels within L1 pixel
            hits = p.contains_points(points)
            mask = hits.reshape(msi_clim.shape)

            # average MSI within pixel
            return np.nanmean(msi_clim[mask])

    def read_msi(self, l1_lon, l1_lat, l1_clon, l1_clat):
        import numpy as np
        import rasterio
        from joblib import Parallel, delayed
        import time
        from scipy.interpolate import griddata
        import os
        import pickle
        import sys
        import matplotlib.pyplot as plt
        from matplotlib.path import Path

        if self.data_type==0:
            # This function collects MSI data on a grid
            # and interpolates an average reflectance to each l1 pixel

            grid_spacing = 0.05    # what is the best choice here?

            save_file = 'msi_clim_'+str(grid_spacing).split('.')[0]+'res'+str(grid_spacing).split('.')[1]+'.pkl'
            if save_file not in os.listdir():
                print('Generating MSI Climatology file ', save_file)

                # set up grid on which to collect msi climatology data
                lon_min, lon_max, lat_min, lat_max = -180.0, 180.0, -90.0, 90.0
                lon = np.arange(lon_min, lon_max, grid_spacing)
                lat = np.arange(lat_min, lat_max, grid_spacing)
                lon,lat = np.meshgrid(lon, lat)

                msi_clim = np.zeros(lon.shape)
                #c=0
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
                        #c+=1
                        msi_lon, msi_lat = np.meshgrid(np.linspace(min_lon, max_lon, width),\
                                                       np.linspace(min_lat, max_lat, height))
                        msi = griddata((msi_lon.ravel(), msi_lat[::-1].ravel()), img.ravel(), (lon, lat))
                        msi[np.isnan(msi)] = 0.0
                        msi[msi_clim!=0] = 0.0          # no double registration
                        msi_clim = msi_clim + msi
                        #if c>1:
                        #    break
                pkl_file = open(save_file, 'wb')
                pickle.dump([lon, lat, msi_clim], pkl_file)
                pkl_file.close()

                # save image
                plt.pcolormesh(lon, lat, msi_clim)
                plt.xlabel('lon')
                plt.ylabel('lat')
                plt.title('MSI gridded climatology')
                plt.savefig(save_file.split('.pkl')[0]+'.png', dpi=600)

            else:
                print('Read MSI Climatology file ', save_file)
                pkl_file = open(save_file, 'rb')
                data = pickle.load(pkl_file)
                lon = data[0]; lat = data[1]; msi_clim = data[2]
                pkl_file.close()

            msi_ref_file = self.l1_name+'_msi_ref.pkl'
            if msi_ref_file not in os.listdir():
                print('Colocate L1 and MSI')

                # center points of the gridded msi data
                lo, la = lon + grid_spacing/2, lat + grid_spacing/2 
                lo, la = lo.flatten(), la.flatten()
                points = np.vstack((lo, la)).T

                # throw out l1 fill values
                l1_clon[np.abs(l1_clon)>180] = np.nan
                l1_clat[np.abs(l1_clat)>90] = np.nan
                l1_lon[np.abs(l1_lon)>180] = np.nan
                l1_lat[np.abs(l1_lat)>90] = np.nan
                # reshape for pixel generation
                l1alt, l1act = l1_lon.shape[0],l1_lon.shape[1]
                l1_lon = l1_lon.reshape(l1_lon.shape[0]*l1_lon.shape[1])
                l1_lat = l1_lat.reshape(l1_lat.shape[0]*l1_lat.shape[1])
                l1_clon = l1_clon.reshape((l1_clon.shape[0], l1_clon.shape[1]*l1_clon.shape[2]))
                l1_clat = l1_clat.reshape((l1_clat.shape[0], l1_clat.shape[1]*l1_clat.shape[2]))

                st = time.time()
                msi_ref = Parallel(n_jobs=6)(\
                     delayed(self.av_msi_per_l1)\
                     (l1_lon[i], msi_clim, points, l1_clon[0,i], l1_clat[0,i], l1_clon[1,i], l1_clat[1,i], l1_clon[2,i], l1_clat[2,i], l1_clon[3,i], l1_clat[3,i])\
                     for i in range(l1_lon.shape[0]))
                msi_ref = np.array(msi_ref).reshape(l1alt, l1act)

                #for i in range(l1_lon.shape[0]):
                #    if np.isnan(l1_lon[i]) or np.isnan(l1_lat[i]):
                #        continue
                #    else:
                #        c1 = (l1_clon[0,i], l1_clat[0,i])
                #        c2 = (l1_clon[1,i], l1_clat[1,i])
                #        c3 = (l1_clon[2,i], l1_clat[2,i])
                #        c4 = (l1_clon[3,i], l1_clat[3,i])
                #        #mask = Parallel(n_jobs=2)(delayed(self.mask_hits)(points, c1, c2, c3, c4, lon.shape))
                #        mask = self.mask_hits(points, c1, c2, c3, c4, lon.shape)
                #        msi_ref[i] = np.nanmean(msi_clim[mask])
                #msi_ref=np.array(msi_ref)
                #msi_ref.reshape(lon.shape)

                print('Finished after', np.round(time.time()-st, 2), ' sec')
                pkl_file = open(msi_ref_file, 'wb')
                pickle.dump([msi_ref], pkl_file)
                pkl_file.close()
            else:
                print('Read colocated MSI reflectance from ', msi_ref_file)
                pkl_file = open(msi_ref_file, 'rb')
                data = pickle.load(pkl_file)
                #msi_ref = data[0]
                msi_ref = np.array(data[0]).reshape(l1_lon.shape[0],l1_lon.shape[1])
                pkl_file.close()

            #plt.pcolormesh(lon, lat, msi_clim, vmin=0, vmax=1)
            #plt.scatter(l1_lon, l1_lat, c=np.array(msi_ref).reshape(l1_lon.shape[0],l1_lon.shape[1]), vmin=0, vmax=1)
            #plt.ylim(-90,90);plt.xlim(-180,180)
            #plt.savefig(msi_ref_file.split('.pkl')[0]+'.png', dpi=300)
            #plt.show()

        else:
            msi_ref = np.zeros(self.prefilter.shape)
            # TODO implement msi reader for measurement data

        return msi_ref
      
    def read_l1(self):
        from netCDF4 import Dataset
        import sys
        import numpy as np
        import ephem
        import datetime

        sza, alb, lon, lat, xch4_true, cf, l1_ref, clon, clat = [],[],[],[],[],[],[],[],[]

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

                clat_tmp = d.groups['Level1'].variables['CornerLatitudes'][:,it0:itf,ix0:ixf].T.squeeze()
                clat_tmp = np.pad(clat_tmp, [(it0,0),(0,0)], 'constant', constant_values=(np.nan))
                clat.append( clat_tmp )
                clon_tmp = d.groups['Level1'].variables['CornerLongitudes'][:,it0:itf,ix0:ixf].T.squeeze()
                clon_tmp = np.pad(clon_tmp, [(it0,0),(0,0)], 'constant', constant_values=(np.nan))
                clon.append( clon_tmp )

                #xch4_tmp = d.groups['Profile'].variables['CH4_ProxyMixingRatio'][0,it0:itf,ix0:ixf].T
                #xch4_tmp = xch4_tmp[:].T.squeeze()
                #xch4_true.append( np.pad(xch4_tmp, (it0,0), 'constant', constant_values=(np.nan)) )

                cf.append(self.get_cloud_fraction(self.truth_path, f, ix0, ixf))

            return np.array(sza).T, np.array(alb).T, np.array(l1_ref).T,\
                     np.array(cf).T, np.array(lon).T, np.array(lat).T,\
                     np.array(clon).T, np.array(clat).T #, np.array(xch4_true).T

        else:
            for f in self.l1_bundle:
                d = Dataset(f)
               
                if len(d.groups) < 3:  # this is an intermediate/pre-L1B file
                    # time and date not in file, get it from file name
                    date = f.split('_')[6].split('T')[0]
                    time = f.split('_')[6].split('T')[1]
                    y = int(date[:4])
                    mo = int(date[4:6])
                    da = int(date[6:8])
                    h = int(time[:2])
                    mi = int(time[2:4])
                    s = int(time[4:6])
                    t = datetime.datetime(y, mo, da, h, mi, s)
 
                    lon_tmp = np.array(d.groups['Geolocation'].variables['Longitude'][:])
                    lon.append(lon_tmp)
                    lat_tmp = np.array(d.groups['Geolocation'].variables['Latitude'][:])
                    lat.append(lat_tmp)

                    # use a single SZA value for entire L1 scene
                    sza_tmp = np.ones(lon_tmp.shape)
                    obs = ephem.Observer()
                    obs.lat = str(np.mean(lat))
                    obs.lon = str(np.mean(lon))
                    obs.date = t
                    sun = ephem.Sun(obs)
                    sun.compute(obs)
                    sza_tmp = sza_tmp * (90. - float(sun.alt) * 180. / np.pi)

                    # if pixel-by-pixel SZA desired, use this:
                    # (takes ~3 sec for 30 sec granule vs ~3 msec for approximation above)
                    #sza_tmp = np.zeros(lon_tmp.shape)
                    #for i in range(sza_tmp.shape[0]):
                    #    for j in range(sza_tmp.shape[1]):
                    #        obs = ephem.Observer()
                    #        obs.lat = str(lat[-1,-1])
                    #        obs.lon = str(lon[-1,-1])
                    #        obs.date = t
                    #        sun = ephem.Sun(obs)
                    #        sun.compute(obs)
                    #        sza_tmp[i,j] = 90. - float(sun.alt) * 180. / np.pi

                    sza.append(sza)
                    rad = np.array(d.groups['Band1'].variables['Radiance'][:])

                    # calculate reflectance spectra
                    I0_CH4 = 1.963e14  # approx I0 @ 1621 nm [photons/s/cm2/nm] 
                                       # (from chance_jpl_mrg_hitran_200-16665nm.nc)
                    l1_ref.append( np.pi * rad\
                                   / (np.cos(np.pi * sza_tmp / 180.) * I0_CH4))

                    print(l1_ref)
                    print(np.array(l1_ref).shape)
                    
                    return np.array(sza), np.array(l1_ref), np.array(lon), np.array(lat)

                #else:  this is a L1B file
                # TODO add reader

           
            

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


