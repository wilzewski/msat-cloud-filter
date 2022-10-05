# -*- coding: utf-8 -*-
# Cloud filter for MethaneAIR/MethaneSAT
# Jonas Wilzewski (jwilzewski@g.harvard.edu)

class cloudFilter(object):

    def __init__(self, l1, msi, glint=False, osse=False, truth_path=None):

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
            self.l1_albedo = np.zeros(filter_shape)
            self.msi_albedo = np.zeros(filter_shape)
            self.cloud_truth = np.zeros(filter_shape)
            self.truth_path = truth_path
            
            if truth_path==None:
                print('Provide truth_path while using OSSE data'); sys.exit()

        else:
            # MSAT/MAIR data
            self.data_type = 1
            # TODO fill in initialization for MAIR data
            
    def get_prefilter(self):
        import matplotlib.pyplot as plt
        import numpy as np

        print('read_l1()')
        sza, alb, ref, cf, lon, lat = self.read_l1()
        
    def read_l1(self):
        from netCDF4 import Dataset
        import sys
        import numpy as np

        sza, alb, lon, lat, xch4_true, cf, ref = [],[],[],[],[],[],[]
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
                ref.append( (np.pi*np.array(rad_tmp).T / 
                              (np.cos(np.pi*np.array(sza_tmp)/180.)*I0_CH4)).T )

                alb_tmp = d.groups['OptProp_Band1'].variables['BRDF_KernelAmplitude_isotr'][:,it0:itf,ix0:ixf].T
                alb_tmp = np.mean(alb_tmp,axis=2).T.squeeze()   # why average?
                alb.append( np.pad(alb_tmp, (it0,0), 'constant', constant_values=(np.nan))  )

                lat_tmp = d.groups['Level1'].variables['Latitude'][0,it0:itf,ix0:ixf].T.squeeze()
                lat.append( np.pad(lat_tmp, (it0,0), 'constant', constant_values=(np.nan)) )
                lon_tmp= d.groups['Level1'].variables['Longitude'][0,it0:itf,ix0:ixf].T.squeeze()
                lon.append( np.pad(lon_tmp, (it0,0), 'constant', constant_values=(np.nan)) )

                #xch4_tmp = d.groups['Profile'].variables['CH4_ProxyMixingRatio'][0,it0:itf,ix0:ixf].T
                #xch4_tmp = xch4_tmp[:].T.squeeze()
                #xch4_true.append( np.pad(xch4_tmp, (it0,0), 'constant', constant_values=(np.nan)) )

                cf.append(self.get_cloud_fraction(self.truth_path, f, ix0, ixf))

            return np.array(sza).T, np.array(alb).T, np.array(ref).T, np.array(cf).T, np.array(lon).T, np.array(lat).T#, np.array(xch4_true).T

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

        return np.array(cloud_fraction)

    def read_msi(self):
        return 0


