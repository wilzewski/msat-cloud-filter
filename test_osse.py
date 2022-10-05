from cloudFilter import cloudFilter

l1 = '/scratch/sao_atmos/ahsouri/Retrieval/methanesat_osses/L1/Global_no_airglow/MethaneSAT_20210830.0_05h_l2prof_l1.nc_subset'

msi = '/scratch/sao_atmos/jwilzews/DATA/MSI/MSI_clim'

truth = '/scratch/sao_atmos/ahsouri/Retrieval/methanesat_osses/profile_prep/geos-fp-profile/airglow_extended/'

fobj = cloudFilter(l1, msi, osse=True, truth_path=truth)

fobj.get_prefilter()
