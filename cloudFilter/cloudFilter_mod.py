# -*- coding: utf-8 -*-
# Cloud filter for MethaneAIR/MethaneSAT
# Jonas Wilzewski (jwilzewski@g.harvard.edu)

class cloudFilter(object):

    def __init__(self, l1, msi, l2_ch4=None, l2_o2=None, l2_h2o=None, glint=False, osse=False):

        from netCDF4 import Dataset
        from collections import deque
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
            l2_* (string): path to L2 data file or data directory for respective retrieval
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
            if len(self.l1_bundle) == 0 : # is this a top level dir w/ many OSSE orbits?
                self.l1istopdir = True
                self.l1_orbits = sorted(glob.glob(l1 + '/*subset'))
                self.l1_bundle = deque()
                self.l1_name = deque()
                for i in range(len(self.l1_orbits)):
                    self.l1_bundle.append(sorted(glob.glob(self.l1_orbits[i] + '/*.nc')))
                    self.l1_name.append(self.l1_orbits[i].split('.')[1].split('.')[0])
                self.l1_bundle = np.array(self.l1_bundle)
                self.l1_name = np.array(self.l1_name)
                if len(self.l1_orbits) == 0 :
                    print('Provide L1 directory containing orbit directories or netCDF data files')
                    sys.exit()
            else:
                self.l1istopdir = False
                self.l1_name = l1.split('.')[1].split('.')[0]
                self.l1_orbits = []
        else:
            # l1 is one file
            self.l1istopdir = False
            self.l1_bundle = []
            self.l1_orbits = []
            self.l1_bundle.append(os.path.abspath(l1))
            if self.data_type == 0:
                self.l1_name = l1.split('/')[-2]
            else:
                self.l1_name = l1.split('/')[-1].split('.nc')[0]

        # Initialize
        if osse:
            if self.l1istopdir:
                # set filter to max dimension found in the orbits:
                alt, act, ch = 0, 0, 0
                for i in range(self.l1_bundle.shape[0]):
                    if Dataset(self.l1_bundle[i,0]).dimensions['jmx'].size > alt:
                        alt = Dataset(self.l1_bundle[i,0]).dimensions['jmx'].size
                    if Dataset(self.l1_bundle[i,0]).dimensions['imx'].size > act:
                        act = Dataset(self.l1_bundle[i,0]).dimensions['imx'].size
                # spectral channel length is not expected to vary in OSSE:
                ch = Dataset(self.l1_bundle[0,0]).dimensions['wmx_2'].size

                filter_shape = (self.l1_bundle.shape[0],alt,act)
                filter_shape_ref = (self.l1_bundle.shape[0],ch,alt,act)
                self.cloud_truth = np.zeros(filter_shape)
            else:
                filter_shape = (Dataset(self.l1_bundle[0]).dimensions['jmx'].size,
                             Dataset(self.l1_bundle[0]).dimensions['imx'].size)
                filter_shape_ref = (Dataset(self.l1_bundle[0]).dimensions['wmx_2'].size,
                             Dataset(self.l1_bundle[0]).dimensions['jmx'].size,
                             Dataset(self.l1_bundle[0]).dimensions['imx'].size)
                self.cloud_truth = np.zeros(filter_shape)
        else:
            filter_shape = (Dataset(self.l1_bundle[0]).dimensions['x'].size,
                         Dataset(self.l1_bundle[0]).dimensions['y'].size)
            filter_shape_ref = (Dataset(self.l1_bundle[0]).dimensions['w1'].size,
                         Dataset(self.l1_bundle[0]).dimensions['x'].size,
                         Dataset(self.l1_bundle[0]).dimensions['y'].size)

        self.prefilter = np.zeros(filter_shape)
        self.prefilter_class_ratio = 0   # ratio cloudy/clear scenes in osse
        self.binary_cloud_fraction_thresh = 0.1
        self.training_imbalance_thresh = 2   # maximum allowed class imbalance in training data
        self.quality_flag = np.zeros(filter_shape)
        self.postfilter = np.zeros(filter_shape)
        self.l1_ref = np.zeros(filter_shape_ref)
        self.l1_valid_mask = np.zeros(self.l1_ref.shape)
        self.l1_wav = np.zeros(filter_shape)
        self.l1_sza = np.zeros(filter_shape)
        self.l1_alb = np.zeros(filter_shape)
        self.msi_ref = np.zeros(filter_shape)
        self.msi_raw = np.zeros(filter_shape)
        self.msi_path = msi
        self.prefilter_model = 0
        self.prefilter_scaler = 0
        self.postfilter_model = 0
        self.postfilter_scaler = 0
        self.yymmdd = 20220101

    def prior_vcd_filter(self, vcd_retr, vcd_prior):
        return vcd_prior - vcd_retr
    
    def vcd_diff_filter(self, vcd_retr1, vcd_retr2):
        #return np.abs(1 - vcd_retr1/vcd_retr2)
        return vcd_retr1/vcd_retr2
    
    def msi_filter(self, l1_ref, msi_ref):
        return l1_ref - msi_ref

    def load_svm_training_data(self):
        import numpy as np
        import scipy.io as sio
        import os

        mat_path = '/scratch/sao_atmos/jwilzews/DATA/OSSE/mat_from_Amir/'

        # variables contained in msi_clim['MSI_clim'], true_cf['Z'], ch4_ref['ref_ch4s_land']
        true_cf = sio.loadmat(os.path.join(mat_path, 'true_cf.mat'))
        ch4 = sio.loadmat(os.path.join(mat_path, 'CH4_VCD.mat'))
        co2 = sio.loadmat(os.path.join(mat_path, 'CO2_VCD.mat'))
        h2o = sio.loadmat(os.path.join(mat_path, 'H2O_VCD.mat'))
        msi_clim = sio.loadmat(os.path.join(mat_path, 'msi_clim_global_osse.mat'))
        ch4_ref = sio.loadmat(os.path.join(mat_path, 'truth_ref_ch4_ocean_land.mat'))
    
        co2_vcd_prior = co2['CO2_true']
        co2_vcd_retr = co2['CO2_ret']
    
        ch4_vcd_prior = ch4['CH4_true']
        ch4_vcd_retr = ch4['CH4_ret']
    
        h2o_retr_1 = h2o['H2O_ret_ch4']
        h2o_retr_2 = h2o['H2O_ret_o2']
    
        # add zero cloud fraction to variable where there are nans:
        true_cf = np.nan_to_num(true_cf['Z'], copy=True, nan=0.0, posinf=None, neginf=None)
    
        ch4_filter = self.prior_vcd_filter(ch4_vcd_retr, ch4_vcd_prior)
        co2_filter = self.prior_vcd_filter(co2_vcd_retr, co2_vcd_prior)
        h2o_diff_filter = self.vcd_diff_filter(h2o_retr_1, h2o_retr_2)
        ref_filter = self.msi_filter(ch4_ref['ref_ch4s_land'], msi_clim['MSI_clim'])
    
        return co2_filter.flatten()/1e21, ch4_filter.flatten()/1e21, ref_filter.flatten(), h2o_diff_filter.flatten(), true_cf.flatten()

    def get_filter(self):
        import numpy as np
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn import preprocessing
        import joblib

        import sys
        import os
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib as mpl

        if self.data_type == 0:  # if osse data load training data and get svm model

            v1, v2, v3, v4, y = self.load_svm_training_data()
            # remove nans:
            y = y[~np.isnan(v1)]
            v4 = v4[~np.isnan(v1)]
            v3 = v3[~np.isnan(v1)]
            v1 = v1[~np.isnan(v1)]
            v2 = v2[~np.isnan(v2)]

            if len(v1)!= len(v2) or len(v1)!=len(v3) or len(v1)!=len(y) or len(v2)!=len(v3) or len(v4)!=len(v1):
                print('array length mismatch')
                sys.exit()
            if np.isnan(v3).sum()>0 or np.isnan(v1).sum()>0 or np.isnan(v2).sum()>0 or np.isnan(v4).sum()>0:
                print('nan present in array')
                sys.exit()

            # for speed-up and testing:
            v4=v4[22100:30200]
            v3=v3[22100:30200]
            v1=v1[22100:30200]
            v2=v2[22100:30200]
            y = y[22100:30200]
            full_colors = y
            print('Input data dim:', len(v1))

            # feature matrix
            #X = np.array([[v1[i], v2[i], v3[i], v4[i]] for i in range(len(v1))])
            X = np.array([[v1[i], v3[i]] for i in range(len(v1))])

            # scale X so variables have mean=0 and std=1
            #scaler = preprocessing.PowerTransformer(method="yeo-johnson").fit(X)
            scaler = preprocessing.RobustScaler().fit(X)
            X = scaler.transform(X)

            for i in X.T:
                print(np.round(np.nanmean(i),2), np.round(np.nanstd(i),2),
                         np.round(np.nanmin(i),2), np.round(np.nanmax(i),2))

            # target
            # needs to be integer values:
            y = np.array((np.round(y*100,0)).astype(float)) # this makes 100 classes
            # make two classes, cloudy = 1, clear = 0
            for i in range(len(y)):
                if y[i]<0.01:
                    y[i]=float(0)
                else:
                    y[i]=float(1)

            n_sample = len(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            # classifier
            gamma=1
            clf = svm.SVC(kernel='rbf', gamma=gamma)#, class_weight="balanced")

            # Fit Data
            clf.fit(X_train, y_train)

            # Predict labels for test data
            clf_pred = clf.predict(X_test)
            print('Accuracy: ', np.round(balanced_accuracy_score(y_test, clf_pred),2))

            # save support vectors
            print('Saving Support Vectors to svm_model.pkl')
            joblib.dump([clf, scaler], 'svm_model.pkl')
        
            do_plots = False
            if do_plots:
                # plot stuff
                plt.figure()
                plt.scatter(
                    X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired_r, edgecolor="k", s=20
                )
                ## Circle out the test data
                #plt.scatter(
                #    X_test[:, 0], X_test[:, 1], s=10, facecolors="black", zorder=10, edgecolor="k"
                #)

                plt.axis("tight")
                x_min = X[:, 0].min()
                x_max = X[:, 0].max()
                y_min = X[:, 1].min()
                y_max = X[:, 1].max()

                XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
                minZ, maxZ = min(Z), max(Z)

                # Put the result into a color plot
                Z = Z.reshape(XX.shape)

                plt.pcolormesh(XX, YY, Z>0, cmap=plt.cm.Paired_r)
                plt.contour(
                    XX,
                    YY,
                    Z,
                    colors=["k", "k", "k"],
                    linestyles=["--", "-", "--"],
                    levels=[-0.5,0, 0.5],
                )
                plt.ylabel('L1 Refl - MSI Refl [Normalized]')
                plt.xlabel('CO2 VCD Prior - Retr [Normalized]')
                #plt.ylabel('CH4 VCD Prior - Retr [Normalized]')
                #plt.ylabel('H2O (CH4) / H2O (O2) [Normalized]')
                plt.title('Cloud Fraction Distribution in Filter Space\ngamma='+str(gamma))
                #plt.savefig('2d_ch4-vs-msi.png')
                plt.show()

                f, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(7,10), sharex=True)

                ymin, ymax = -1,1
                xmin, xmax = -1,1

                cmap=mpl.cm.viridis
                cmap.set_under('red')

                sc1 = ax1.scatter(X[:,0], X[:,1], alpha=0.5, c=full_colors, cmap=cmap, vmin=0.01)
                self.add_cbar_title_axis_extend_min(f, ax1, sc1, 'True Cloud Fraction', '')
                ax1.set_title('Truth')

                cmap=mpl.colors.ListedColormap(['red', 'green'])

                sc2 = ax2.scatter(X_train[:,0],X_train[:,1], alpha=0.5,c=y_train, cmap=cmap)
                ax2.set_ylabel('L1 Refl - MSI Refl [Normalized]')
                #ax2.set_ylabel('CH4 VCD Prior - Retr [Normalized]')
                #ax2.set_ylabel('H2O (CH4) / H2O (O2) [Normalized]')
                self.add_cbar_title_axis_binary(f, ax2, sc2, 'Clear', 'Cloudy', 'Training Data')

                sc3 = ax3.scatter(X_test[:,0],X_test[:,1], alpha=0.5,c=clf_pred, cmap=cmap)
                #ax3.set_xlabel('CH4 VCD Prior - Retr [Normalized]')
                ax3.set_xlabel('CO2 VCD Prior - Retr [Normalized]')
                self.add_cbar_title_axis_binary(f, ax3, sc3, 'Clear', 'Cloudy', 'Test Data - SVM Classification')
                #ax3.text(2, -0.4, 'Accuracy: '+str(np.round(accuracy_score(y_test, clf_pred),2)))
                ax3.text(-3, 2, 'Accuracy: '+str(np.round(accuracy_score(y_test, clf_pred),2)))
                #plt.savefig('trainig_ch4_msi.png')
                plt.show()
                
        else: # load model and apply to L2 data 
             
            # read support vectors
            print('Reading Support Vectors from svm_model.pkl')
            self.postfilter_model = joblib.load('svm_model.pkl')[0]
            self.postfilter_scaler = joblib.load('svm_model.pkl')[1]
       
    def apply_filter(self, approach=None):
        print('apply_filter')
        #read_l1()
        #read_msi()
        #read_l2()
        #interpolate_all_l2_to_ch4grid()

         


    def remove_clusters(self, prefilter, min_pix_size):
        import numpy as np
        from skimage.measure import label
        # min_pix_size: minimum of connected pixels to be retained as cloud
        # label connected pixel-areas
        prefilter, num = label(prefilter, connectivity=2, return_num=True)

        for i in np.arange(1,num+1):
            n_pix = np.count_nonzero(prefilter == i)
            if n_pix < min_pix_size:   # remove cloud label if too few pixels
                prefilter[prefilter == i] = 0
        prefilter[prefilter != 0] = 1 # reset labels

        return prefilter

    def apply_prefilter(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler,RobustScaler
        import pickle
        import sys
        import os
        from scipy import ndimage

        print('read_l1()')
        self.l1_sza, self.l1_ref, l1_lon, l1_lat = self.read_l1()

        if self.l1_name+'_msi_B11.pkl' not in os.listdir():
            print('read_msi_B11()')
            self.msi_ref = self.read_msi_B11(l1_lon, l1_lat)

            pkl_file = open(self.l1_name+'_msi_B11.pkl', 'wb')
            pickle.dump(self.msi_ref, pkl_file)
            pkl_file.close()
        else:
            print('read msi_ref from file')
            pkl_file = open(self.l1_name+'_msi_B11.pkl', 'rb')
            self.msi_ref = pickle.load(pkl_file)
            pkl_file.close()

        self.msi_ref = ndimage.maximum_filter(self.msi_ref, size=(1,200,50))

        X, order = self.prepare_prefilter_data()

        #scaler = StandardScaler()
        scaler = RobustScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        #X = self.prefilter_scaler.transform(X)

        pred = self.prefilter_model.predict(X)

        tmp = np.zeros(self.prefilter.shape)
        tmp[self.l1_valid_mask] = pred
        
        self.prefilter = tmp

        # remove small clouds/misclassifications
        self.prefilter = self.remove_clusters(self.prefilter, 500)
        # ratio of flagged clouds to total pixels
        r_flagged = np.sum(self.prefilter)/np.count_nonzero(~np.isnan(self.l1_ref[630,:,:]))
        print('Fraction of potentially cloudy pixels: ', np.round(r_flagged, 2))

        # if substantial amount of clouds present, check for shadows:
        #if any(self.prefilter.flatten()):
        #if r_flagged > 0.01:
        #    # for now just filter out dark scenes:
        #    ind = np.where(np.nanmean(self.l1_ref, axis=0) < 0.1)
        #    self.prefilter[ind[0], ind[1]] = 1
        #    self.prefilter = self.remove_clusters(self.prefilter, 500)
            
        r_flagged = np.sum(self.prefilter)/np.count_nonzero(~np.isnan(self.l1_ref[630,:,:]))
        print('Fraction of cloudy+shadowy pixels: ', np.round(r_flagged, 2))

        plt.imshow(self.l1_ref[630,:,:], aspect='auto', interpolation='none')
        cb = plt.colorbar();cb.set_label('Reflectance')
        masked_pixels = np.ma.masked_where(self.prefilter == 0, self.prefilter)
        plt.imshow(masked_pixels, aspect='auto', cmap='autumn', alpha=0.5)
        plt.xlabel('Along Track Index')
        plt.ylabel('Across Track Index')
        plt.savefig(self.l1_name+'_prefilter.png', dpi=300)
        plt.show()
        plt.imshow(self.l1_ref[630,:,:], aspect='auto', interpolation='none')
        cb = plt.colorbar();cb.set_label('Reflectance')
        plt.imshow(self.msi_ref, aspect='auto', alpha=0.5)
        plt.xlabel('Along Track Index')
        plt.ylabel('Across Track Index')
        plt.savefig(self.l1_name+'_vs_msi_filtered.png', dpi=300)

        return self.prefilter

    def read_msi_B11(self, l1_lon, l1_lat):
        # read MSI jp2 files
        import numpy as np
        import os
        import rasterio
        import utm
        import cv2
        from shapely.geometry import Polygon
        import sys
        import matplotlib.pyplot as plt
        import time
        
        intersect_box = []
        msi_date_intsec = []

        # loop through MSI files:
        for f in sorted(os.listdir(self.msi_path)):
            #if 'T12TVK' not in f:
            #    continue

            file_size = os.path.getsize(self.msi_path+'/'+f)
            # read MSI information
            src = rasterio.open(self.msi_path+'/'+f,driver='JP2OpenJPEG')
            # UTM zones
            zones = (int(str(src.crs)[-2::]))
            out_trans = src.transform
            # get the boundaries
            width = src.width
            height = src.height

            temp =  out_trans * (0,0)
            corner1 = np.array(utm.to_latlon(temp[0],temp[1],int(zones),'T'))
            temp =  out_trans * (height,width)
            corner4 = np.array(utm.to_latlon(temp[0],temp[1],int(zones),'T') )

            # make polygons for both l1 and msi images based on their corners
            p_msi = Polygon([(corner1[1],corner4[0]), (corner4[1],corner4[0]), (corner4[1],corner1[0]),                              (corner1[1],corner1[0]), (corner1[1],corner4[0])])

            if l1_lon.shape[0]==1: # L1 is a single file
                p_l1 = Polygon([(np.min(l1_lon), np.min(l1_lat)), 
                                  (np.max(l1_lon), np.min(l1_lat)), (np.max(l1_lon), np.max(l1_lat)), 
                                  (np.min(l1_lon), np.max(l1_lat)), (np.min(l1_lon), np.min(l1_lat))])

            else:   # TODO add for L1 list/dir
                print('l1 data are list/from directory - add code here');sys.exit()

            if (p_msi.intersects(p_l1)) and file_size>15096676:
                intersect_box.append(self.msi_path+'/'+f)
                msi_date_intsec.append(float(f.split('_')[-2].split('T')[0]))

        if (not intersect_box):
            print('No MSI files being relevant to the targeted location/time were found, please fetch more MSI data')
            return np.zeros(self.prefilter.shape)
            # TODO add an option to read from MSI climatology here

        else:    # MSI images overlapping the L1 field were found
            # which MSI image is temporaly closest to L1?
            dist_date = np.abs(np.array(msi_date_intsec) - float(self.yyyymmdd))
            dist_date_sorted = sorted(dist_date)
            counter = 0
            # finding the most relevant MSI images based on date
            index_chosen_sorted = []
            for i in range(np.size(dist_date_sorted)):
                j = np.where(dist_date == dist_date_sorted[i])[0]
                for p in range(np.size(j)):
                    if counter>10:
                        break
                    index_chosen_sorted.append(j[p])
                    counter = counter + 1

            msi_grays = []
            lat_msis = []
            lon_msis = []
            # loop over selected MSI images
            for index_bundle in range(len(index_chosen_sorted)):
                # read MSI
                src = rasterio.open(intersect_box[index_chosen_sorted[index_bundle]],driver='JP2OpenJPEG')
                # UTM zone
                zones = (int(str(src.crs)[-2::]))
                # the transformation from i,j to E and N
                out_trans = src.transform
                msi_img = src.read(1) # grayscale image
                print('Reading MSI image ' +  intersect_box[index_chosen_sorted[index_bundle]])
              
                # i,j to E and N
                E_msi = np.zeros_like(msi_img)*np.nan
                N_msi = np.zeros_like(msi_img)*np.nan
                t0 = time.time()
                for i in range(np.shape(E_msi)[0]):
                    for j in range(np.shape(E_msi)[1]):
                        temp = out_trans * (i,j)
                        E_msi[i,j] = temp[0] 
                        N_msi[i,j] = temp[1]
                print('Finished loop in ', np.round(time.time()-t0,2), ' sec')

                E_msi = np.float32(E_msi)
                N_msi = np.float32(N_msi)
                
                # E and N to lon and lat
                temp = np.array(utm.to_latlon(E_msi.flatten(),N_msi.flatten(),int(zones),'T'))
                temp2 = np.reshape(temp,(2,np.shape(msi_img)[0],np.shape(msi_img)[1]))

                lat_msi = np.squeeze(temp2[0,:,:])
                lon_msi = np.squeeze(temp2[1,:,:])

                msi_gray = np.array(msi_img, dtype='uint16').astype('float32')

                msi_grays.append(np.transpose(msi_gray))
                lat_msis.append(lat_msi)
                lon_msis.append(lon_msi)

        print('Interpolating MSI images onto L1 grid')
        for msi_ind in range(len(msi_grays)): # looping over selected MSI images
            r = self.cutter(msi_grays[msi_ind],lat_msis[msi_ind],lon_msis[msi_ind], l1_lon, l1_lat)
            if msi_ind == 0:
                final_msi = np.zeros_like(r)
            r[final_msi != 0.0] = 0.0
            final_msi = final_msi + r

        # a scaling factor should be applied to make ref physical
        self.msi_raw = final_msi/10000.0
        
        #r = final_msi

        # normalizing between 0 and 1 based on min/max
        #msi_ref = cv2.normalize(r,np.zeros(r.shape, np.double),0.0,1.0,cv2.NORM_MINMAX)
        # histogram equalization for enhancing image contrast
        #clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
        #self.master = clahe.apply(np.uint8(msi_ref*255))
        # uncomment this to run without contrast enhancement
        #msi_ref = np.uint8(msi_ref*255)

        return self.msi_raw

    def cutter(self,rad,lat,lon, l1_lon, l1_lat):
        '''
        subset the large msi/landsat data based on min/max lons/lats
        ARGS:
           rad(float) : radiance
           lat(float) : latitude
           lon(float) : longitude
        OUT:
           rad(float) : radiance
        ''' 
        import numpy as np
        from scipy.interpolate import griddata 

        # find the range in lon and lat of the slave image
        lon_range = np.array([np.min(l1_lon),np.max(l1_lon)])
        lat_range = np.array([np.min(l1_lat),np.max(l1_lat)])
        # create a mask
        mask_lon = (lon >= lon_range[0]) & (lon <= lon_range[1])
        mask_lat = (lat >= lat_range[0]) & (lat <= lat_range[1])
        # mask it, f**king, mask it
        rad = rad [ mask_lon & mask_lat ]
        lat = lat [ mask_lon & mask_lat ]
        lon = lon [ mask_lon & mask_lat ]
        
        # regrid the master into the slave lats/lons
        points = np.zeros((np.size(lat),2))
        points[:,0] = lon.flatten()
        points[:,1] = lat.flatten()
        rad = griddata(points, rad.flatten(), (l1_lon, l1_lat), method='linear')
        
        # masking based on bad data in the slave
        #if self.typesat_master == 0: # no need to mask for O2-CH4 relative correction
        #   pass
        #else:
        #   rad[self.maskslave] = np.nan
        return rad

    def get_prefilter(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import sys
        import os
        import pickle

        if self.data_type==0:
            print('read_l1()')
            if 'osse_l1.pkl' not in os.listdir():
                self.l1_sza, self.l1_alb, self.l1_ref, \
                     self.cloud_truth, l1_lon, l1_lat, l1_clon, l1_clat = self.read_l1()
                pkl_file = open('osse_l1.pkl', 'wb')
                pickle.dump([self.l1_sza, self.l1_alb, self.l1_ref, \
                     self.cloud_truth, l1_lon, l1_lat, l1_clon, l1_clat], pkl_file)
                pkl_file.close()
            else:
                print('...from file osse_l1.pkl')
                pkl_file = open('osse_l1.pkl', 'rb')
                d = pickle.load(pkl_file)
                self.l1_sza, self.l1_alb, self.l1_ref, \
                     self.cloud_truth, l1_lon, l1_lat, l1_clon, l1_clat = d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]
                pkl_file.close()

            print('read_msi_clim()')
            self.msi_ref = self.read_msi_clim(l1_lon, l1_lat, l1_clon, l1_clat)

            # if model not present:
            if 'prefilter_model.pkl' not in os.listdir():
                print('build_prefilter_model()')
                print('Classification threshold in OSSE is', self.binary_cloud_fraction_thresh)
                self.prefilter_model, self.prefilter_scaler = self.build_prefilter_model()
            else:
                print('Read prefilter model from prefilter_model.pkl')
                pkl_file = open('prefilter_model.pkl', 'rb')
                data = pickle.load(pkl_file)
                self.prefilter_model, self.prefilter_scaler = data[0], data[1]
                pkl_file.close()
        else:
            if 'prefilter_model.pkl' not in os.listdir():
                print('Prefilter model not found. Create first (run code on OSSE).')
                sys.exit()
            else:
                print('Read prefilter model from prefilter_model.pkl')
                pkl_file = open('prefilter_model.pkl', 'rb')
                data = pickle.load(pkl_file)
                self.prefilter_model, self.prefilter_scaler = data[0], data[1]
                pkl_file.close()

    def build_prefilter_model(self):
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import cross_val_score
        import sys
        import numpy as np
        import pickle
        import matplotlib.pyplot as plt

        X, y = self.prepare_prefilter_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=1)

        # downsample if necessary
        if int(np.round(1/self.prefilter_class_ratio,0)) > self.training_imbalance_thresh:
            X_train, y_train = self.downsample(X_train, y_train, clear_cloudy_ratio = self.training_imbalance_thresh)

        # feature scaling so variables have mean=0 and std=1
        #scaler = StandardScaler()
        scaler = RobustScaler()
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)

        model = MLPClassifier(max_iter=2000, alpha=1e-5, verbose=True,
            hidden_layer_sizes=(10,10), activation='tanh', 
            learning_rate='constant', solver='lbfgs', random_state=1)
        #model = MLPClassifier(max_iter=800, hidden_layer_sizes=(5,), alpha=1e-7, solver='sgd', random_state=1)
        #scores = cross_val_score(model, X, y, cv=10)
        #print(scores)
        #print("Cross validation scores:\n  %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        ## tune model set up
        #print('Tune model')
        #parameter_space = {
        #'hidden_layer_sizes': [(10,10,), (15,), (20,10,), (25,)],
        #'max_iter': [2000, 3000],
        #'activation': ['logistic', 'relu', 'tanh'],
        #'solver': ['adam', 'lbfgs', 'sgd'],
        #'alpha': [1e-7, 1e-6, 1e-5, 1e-4],
        #'learning_rate': ['constant', 'adaptive']
        #}

        #clf = GridSearchCV(model, param_grid=parameter_space, n_jobs=-1, cv=10)
        #clf.fit(X, y)

        #print(clf.cv_results_, '\n')
        #print(clf.best_estimator_, '\n')
        #print(clf.best_score_, '\n')
        #print(clf.best_params_, '\n')
        #sys.exit()

        # Train model
        model.fit(X_train, y_train)
        # Predict
        pred = model.predict(X_test)
        print(pred)
        #print('Accuracy: ', np.round(accuracy_score(y_test, pred),2))
        print('Accuracy: ', np.round(balanced_accuracy_score(y_test, pred),2))

        correct_clear = np.where((y_test==0) & (pred==0))
        correct_cloud = np.where((y_test==1) & (pred==1))
        false_clear = np.where((y_test==1) & (pred==0))
        false_cloud = np.where((y_test==0) & (pred==1))

        plt.scatter(X_test[:,3][correct_clear], X_test[:,1][correct_clear], label='Correct prediction: clear', color='green', alpha=0.7)
        plt.scatter(X_test[:,3][correct_cloud], X_test[:,1][correct_cloud], label='Correct prediction: cloud', color='lime', marker='s', alpha=0.6)

        plt.scatter(X_test[:,3][false_clear], X_test[:,1][false_clear], label='False prediction: clear', color='red', alpha=0.4)
        plt.scatter(X_test[:,3][false_cloud], X_test[:,1][false_cloud], label='False prediction: cloud', color='deeppink', marker='s', alpha=0.3)
        plt.legend()
        plt.xlabel('scaled continuum reflectance')
        plt.ylabel('scaled MSI reflectance')
        plt.savefig('prefilter_skill1.png', dpi=300)
        plt.show()

        plt.scatter(X_train[:,10], y_train, label='train')
        plt.scatter(X_test[:,10], y_test, label='test')
        plt.scatter(X_test[:,10], pred, facecolors='none', edgecolor='green', label='model')
        plt.legend()
        plt.xlabel('scaled continuum reflectance')
        plt.ylabel('Cloud flag')
        plt.savefig('prefilter_skill2.png', dpi=300)
        plt.show()

        #sys.exit()
        # save model
        save_file = 'prefilter_model.pkl'
        print('Saving prefilter model to ', save_file)
        pkl_file = open(save_file, 'wb')
        pickle.dump([model, scaler], pkl_file)
        pkl_file.close()

        return model, scaler

    # Utility function to report best scores
    def report(self, grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

    def prepare_prefilter_data(self):
        import pickle
        import numpy as np
        import sys
        import matplotlib.pyplot as plt
        import os
        import warnings
        warnings.filterwarnings("ignore")
        # binary_cloud_fraction_thresh sets the OSSE cloud fraction value above which the 
        # model considers a pixel "cloudy" (cloud fractions below are considered "clear")

        if self.l1istopdir:
            # remove orbit dimension from data
            self.l1_sza = self.l1_sza.reshape(-1, self.l1_sza.shape[2])
            self.l1_ref = self.l1_ref.transpose(1,0,2,3).reshape(self.l1_ref.shape[1], -1, self.l1_ref.shape[3])
            self.msi_ref = self.msi_ref.reshape(-1, self.msi_ref.shape[2])
            self.cloud_truth = self.cloud_truth.reshape(-1, self.cloud_truth.shape[2])

        if self.data_type == 1:
            # when running on the MAIR data, l1 fields need to be squeezed
            # to remove leading "1"/nfiles dimension, should do this differently!
            self.l1_sza = self.l1_sza.squeeze()
            self.l1_ref = self.l1_ref.squeeze()
            self.msi_ref = self.msi_ref.squeeze()

        # clean up conditions: high sun, good spectra, no water
        good_data = np.where((self.l1_sza <= 70) &\
                             (np.nanmean(self.l1_ref, axis=0) <= 1)&\
                              # remove all nan spectra (MAIR)
                             (~np.isnan(self.l1_ref).all(axis=0))&\
                             (self.msi_ref > 0.001))   # TODO keep water?
        self.l1_valid_mask = good_data

        msi = self.msi_ref[good_data]
        sza = self.l1_sza[good_data]
        if self.data_type == 0:
            ref = self.l1_ref[:, good_data[0], good_data[1]]
        else:
            # MAIR: cut off wavelengths since qeff deteriorates after index ~700
            ref = self.l1_ref[:700, good_data[0], good_data[1]]

        #plt.plot(np.linspace(0,1,len(ref[:700,0])), ref[:700,100]);plt.show()

        if self.data_type==0:   # OSSE data run
            cf = self.cloud_truth[good_data]

        # spectral sorting order
        if self.data_type == 0:
            sorting_order_file = 'sorting_order_osse.pkl'
            if sorting_order_file not in os.listdir():
                order = self.spectral_sorting(ref, cf)
                print('Saving sorting order to ', sorting_order_file)
                pkl_file = open(sorting_order_file, 'wb')
                pickle.dump(order, pkl_file)
                pkl_file.close()
            else:
                print('Reading sorting order from ', sorting_order_file)
                pkl_file = open(sorting_order_file, 'rb')
                order = pickle.load(pkl_file)
                pkl_file.close()
        else:
            sorting_order_file = 'sorting_order_MAIR.pkl'
            if sorting_order_file not in os.listdir():
                # sorting based on decently bright spectrum, random selection, use clear granule!
                ind = np.where((np.nanmean(ref, axis=0) > 0.3) & (np.nanmean(ref, axis=0) < 0.4))
                order = np.argsort(ref[:,ind[0][0]])
                print('Saving sorting order to ', sorting_order_file)
                pkl_file = open(sorting_order_file, 'wb')
                pickle.dump(order, pkl_file)
                pkl_file.close()
            else:
                print('Reading sorting order from ', sorting_order_file)
                pkl_file = open(sorting_order_file, 'rb')
                order = pickle.load(pkl_file)
                pkl_file.close()


        ref = ref[order, :].T
        X = np.stack([sza, msi], axis=1)
        # append full, sorted reflectance spectra
        #X = np.hstack((X, ref))

        # alternatively, append average sorted spectra:
        nchannels = 10
        r=np.zeros((ref.shape[0],nchannels))
        l = int(ref.shape[1]/nchannels)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r[i,j] = np.nanmean(ref[i,j*l:(j+1)*l])
        X = np.hstack((X, r))

        if self.data_type == 0:
            # make binary label
            thresh = self.binary_cloud_fraction_thresh
            cf[cf>thresh] = 1
            cf[cf<=thresh] = 0
            self.prefilter_class_ratio = len(cf[cf==1]) / len(cf[cf==0])
            print('clear : cloud ratio is ~ ', int(np.round(1/self.prefilter_class_ratio,0)), ': 1')
            return X, cf
        else:
            return X, order

    def downsample(self, X, cf, clear_cloudy_ratio):
        # remove data from the majority cloud fraction class to match
        # the required clear_cloud_ratio
        # return downsampled X and cf
        import numpy as np

        print('Downsampling majority class so ratio is ', clear_cloudy_ratio, ': 1')

        if self.prefilter_class_ratio > 1:
            majority_class = 1 # more clouds
            minority_class = 0
        else:
            majority_class = 0 # more clear scenes
            minority_class = 1

        N_majority = len(cf[cf==majority_class])
        N_minority = len(cf[cf==minority_class])
        # length of downsampled majority class
        if self.prefilter_class_ratio > 1:
            N_majority_ds = N_minority / clear_cloudy_ratio
        else:
            N_majority_ds = N_minority * clear_cloudy_ratio

        ind_major = np.where(cf==majority_class)[0]
        ind_minor = np.where(cf==minority_class)[0]

        # remove random data from majority class:
        ind_major_ds = np.random.choice(ind_major, size=N_majority_ds, replace=False)

        cf_ds = np.hstack((cf[ind_minor], cf[ind_major_ds]))
        X_ds = np.vstack((X[ind_minor,:], X[ind_major_ds,:]))

        return X_ds, cf_ds

    def spectral_sorting(self, ref, cf):
        # approach inspired by https://doi.org/10.1029/2018GL079286
        # find sorting order
        import numpy as np
        import matplotlib.pyplot as plt
        import sys

        ref_0 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]==0.0]), axis=0)
        order = np.argsort(ref_0)
        #f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True)
        #ax1.plot(np.linspace(1598,1676,ref_0.shape[0]), ref_0)
        #ax2.plot(np.linspace(0,ref_0.shape[0],ref_0.shape[0]), ref_0[order])
        #fs = 12
        #ax1.set_xlabel('Wavelength / nm', fontsize=fs)
        #ax1.set_title('Simulated Spectrum', fontsize=fs)
        #ax2.set_title('Sorted by Reflectance', fontsize=fs)
        #ax2.set_xlabel('Spectral Index', fontsize=fs)
        #ax1.set_ylabel('Reflectance', fontsize=fs)
        #plt.savefig('spectral_sorting_osse.png', dpi=300)
        #plt.show()
        #ref_1 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.2 and cf[i]<=0.3]), axis=0)
        #ref_2 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.3 and cf[i]<=0.4]), axis=0)
        #ref_3 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.4 and cf[i]<=0.5]), axis=0)
        #ref_4 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.5 and cf[i]<=0.6]), axis=0)
        #ref_5 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.6 and cf[i]<=0.7]), axis=0)
        #ref_6 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.7 and cf[i]<=0.8]), axis=0)
        #ref_7 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.8 and cf[i]<=0.9]), axis=0)
        #ref_8 = np.nanmean(np.array([ref[:,i] for i in range(ref.shape[1]) if cf[i]>0.9]), axis=0)
        #x = np.linspace(0,ref_0.shape[0],ref_0.shape[0])
        #plt.figure(figsize=(8,5))
        ##plt.plot(x, self.smooth_curve(x, ref_0[order],5), label='cf=0')
        ##plt.plot(x, self.smooth_curve(x, ref_1[order],5), label='0.2<cf<=0.3')
        ##plt.plot(x, self.smooth_curve(x, ref_2[order],5), label='0.3<cf<=0.4')
        ##plt.plot(x, self.smooth_curve(x, ref_3[order],5), label='0.4<cf<=0.5')
        ##plt.plot(x, self.smooth_curve(x, ref_4[order],5), label='0.5<cf<=0.6')
        ##plt.plot(x, self.smooth_curve(x, ref_5[order],5), label='0.6<cf<=0.7')
        #plt.plot(x, self.smooth_curve(x, ref_6[order],5), label='0.7<cf<=0.8')
        #plt.plot(x, self.smooth_curve(x, ref_7[order],5), label='0.8<cf<=0.9')
        #plt.plot(x, self.smooth_curve(x, ref_8[order],5), label='0.9<cf')
        ##plt.xlim(0,500)
        #plt.ylabel('Reflectance', fontsize=fs)
        #plt.xlabel('Spectral Index', fontsize=fs)
        #plt.title('Sorting order (from clear spectra) applied to cloudy scenes in OSSE')
        #plt.legend()
        #plt.savefig('spectral_sorting_osse_cloudy_spectra_sorted_like_clear_ones.png', dpi=300)
        #plt.show();sys.exit()
        return order

    def smooth_curve(self,x,y,o):
        import numpy as np
        p = np.polyfit(x,y,o)
        return np.poly1d(p)(x)

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

    def read_msi_clim(self, l1_lon, l1_lat, l1_clon, l1_clat):
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

            if self.l1istopdir:
                msi_ref = np.zeros(self.prefilter.shape) * np.nan
                for i in range(self.msi_ref.shape[0]):
                    if '0_077' in self.l1_name[i]:
                        continue
                    msi_ref_file = self.l1_name[i]+'_msi_ref.pkl'
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
                        l1alt, l1act = l1_lon.shape[1],l1_lon.shape[2]
                        l1_lo = l1_lon[i,:,:].reshape(l1_lon.shape[1]*l1_lon.shape[2])
                        l1_la = l1_lat[i,:,:].reshape(l1_lat.shape[1]*l1_lat.shape[2])
                        l1_clo = l1_clon[i,:,:,:].reshape((l1_clon.shape[1], l1_clon.shape[2]*l1_clon.shape[3]))
                        l1_cla = l1_clat[i,:,:,:].reshape((l1_clat.shape[1], l1_clat.shape[2]*l1_clat.shape[3]))
        
                        st = time.time()
                        msi_r = Parallel(n_jobs=6)(\
                             delayed(self.av_msi_per_l1)\
                             (l1_lo[j], msi_clim, points, l1_clo[0,j], l1_cla[0,j], l1_clo[1,j], l1_cla[1,j], l1_clo[2,j], l1_cla[2,j], l1_clo[3,j], l1_cla[3,j])\
                             for j in range(l1_lo.shape[0]))
                        print(np.array(msi_r).shape)
                        print(msi_ref[i,:,:].shape)
                        msi_ref[i,:,:] = np.array(msi_r).reshape(l1alt, l1act)
        
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
                        pickle.dump([msi_ref[i,:,:]], pkl_file)
                        pkl_file.close()
                    else:
                        print('Read colocated MSI reflectance from ', msi_ref_file)
                        pkl_file = open(msi_ref_file, 'rb')
                        data = pickle.load(pkl_file)
                        msi_ref[i,:,:] = np.array(data[0]).reshape(l1_lon.shape[1],l1_lon.shape[2])
                        pkl_file.close()

            else:
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
        import matplotlib.pyplot as plt
        import sys
        import numpy as np
        import ephem
        import datetime

        sza, alb, lon, lat, xch4_true, cf,\
                 l1_ref, clon, clat, yyyymmdd = [],[],[],[],[],[],[],[],[],[]

        if self.data_type==0:
            if self.l1istopdir:
                s = np.ones(self.prefilter.shape) * np.nan
                a = np.ones(self.prefilter.shape) * np.nan
                lo = np.ones(self.prefilter.shape) * np.nan
                la = np.ones(self.prefilter.shape) * np.nan
                xch4_t = np.ones(self.prefilter.shape) * np.nan
                ref = np.ones(self.l1_ref.shape) * np.nan
                clo = np.ones((self.prefilter.shape[0], 4, self.prefilter.shape[1], self.prefilter.shape[2])) * np.nan
                cla = np.ones((self.prefilter.shape[0], 4, self.prefilter.shape[1], self.prefilter.shape[2])) * np.nan
                c = np.ones(self.prefilter.shape)*np.nan
                date = np.ones(self.prefilter.shape[0])*np.nan
                for i in range(len(self.l1_orbits)):
                    if '0_077h' in self.l1_orbits[i]:
                        s[i,:,:] = np.ones((self.prefilter.shape[1],self.prefilter.shape[2]))*np.nan
                        continue
                    sza, alb, lon, lat, xch4_true, cf,\
                       l1_ref, clon, clat, yyyymmdd = [],[],[],[],[],[],[],[],[],[]

                    print('\n', self.l1_orbits[i])
                    for f in self.l1_bundle[i]:
                        d = Dataset(f)
                        yyyymmdd.append(np.float(f.split('.')[0].split('_')[-1]))
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

                    # if along-track dimension of orbit is less than expected: fill with nans
                    if itf != self.prefilter.shape[1]:
                        diff = self.prefilter.shape[1] - itf
                        s[i,:,:] = np.pad(np.array(sza).T, ((0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        a[i,:,:] = np.pad(np.array(alb).T, ((0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        lo[i,:,:] = np.pad(np.array(lon).T, ((0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        la[i,:,:] = np.pad(np.array(lat).T, ((0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        #xch4_t[i,:,:] = np.pad(np.array(xch4_true).T, ((0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        ref[i,:,:,:] = np.pad(np.array(l1_ref).T, ((0,0), (0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        clo[i,:,:,:] = np.pad(np.array(clon).T, ((0,0), (0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        cla[i,:,:,:] = np.pad(np.array(clat).T, ((0,0), (0,diff), (0,0)), mode='constant', constant_values=(np.nan,))
                        c[i,:,:] = np.array(cf).T # cloud fraction is already filled to right extent
                    else:
                        s[i,:,:] = np.array(sza).T
                        a[i,:,:] = np.array(alb).T
                        lo[i,:,:] = np.array(lon).T
                        la[i,:,:] = np.array(lat).T
                        ref[i,:,:,:] = np.array(l1_ref).T
                        clo[i,:,:,:] = np.array(clon).T
                        cla[i,:,:,:] = np.array(clat).T
                        c[i,:,:] = np.array(cf).T
                        #xch4_t[i,:,:] = np.array(xch4_true).T
                    date[i] = np.nanmedian(yyyymmdd)
    
                self.yyyymmdd = np.nanmedian(date)

                return s, a, ref,\
                     c, lo, la,\
                     clo, cla #, np.array(xch4_true).T

            else:
                for f in self.l1_bundle:
                    d = Dataset(f)
                    yyyymmdd.append(np.float(f.split('.')[0].split('_')[-1]))
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

                self.yyyymmdd = np.median(yyyymmdd)
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
                    yyyymmdd.append(np.float(date[:4]+date[4:6]+date[6:8]))
 
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

                    sza.append(sza_tmp)
                    rad = np.array(d.groups['Band1'].variables['Radiance'][:])

                    # calculate reflectance spectra
                    I0_CH4 = 1.963e14  # approx I0 @ 1621 nm [photons/s/cm2/nm] 
                                       # (from chance_jpl_mrg_hitran_200-16665nm.nc)
                    l1_ref.append( np.pi * rad\
                                   / (np.cos(np.pi * sza_tmp / 180.) * I0_CH4))

                #else:  this is a L1B file
                # TODO add reader

            self.yyyymmdd = np.median(yyyymmdd)
            return np.array(sza), np.array(l1_ref), np.array(lon), np.array(lat)

    def get_cloud_fraction(self, osse_path, l1_file, ix0, ixf):
        import os
        from netCDF4 import Dataset
        import numpy as np
        import sys

        cloud_fraction = []
        for f in sorted(os.listdir(osse_path)):
            if l1_file.split('h_')[0].split('SAT')[1] in f:
                profile_input = Dataset(os.path.join(osse_path,f))
                cf = profile_input['SupportingData/CloudFraction'][:,ix0:ixf].squeeze()
                cloud_fraction.append(cf)
        # check status and catch OSSE orbits with fewer along-track data:
        if self.l1istopdir:
            if len(cloud_fraction[0]) != self.prefilter.shape[1]:
                diff = self.prefilter.shape[1] - len(cloud_fraction[0])
                if diff > 0.8 * self.prefilter.shape[1]:
                    print('L1 has substantially less along-track data than expected');sys.exit()
                else:  # pad shorter OSSE orbits with nans
                    for j in range(len(cloud_fraction)):
                        cloud_fraction[j] = np.pad(cloud_fraction[j], (0,diff), mode='constant', constant_values=(np.nan,))
        else:
            if len(cloud_fraction[0]) != self.prefilter.shape[0]:
                print('OSSE Truth data wrong dimension');sys.exit()

        return np.array(cloud_fraction).squeeze()

    # helper fuctions for plotting
    def add_cbar_title_axis(self, f, ax, im, label, title):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='1%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical', label=label)
        ax.set_title(title)
    
    def add_cbar_title_axis_binary(self, f, ax, im, label1, label2, title):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='1%', pad=0.05)
        cbar = f.colorbar(im, cax=cax, orientation='vertical', label='', ticks=[0, 1])
        cbar.ax.set_yticklabels([label1, label2])
        ax.set_title(title)
    
    
    def add_cbar_title_axis_extend_min(self, f, ax, im, label, title):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='1%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical', label=label, extend='min')
        ax.set_title(title)

