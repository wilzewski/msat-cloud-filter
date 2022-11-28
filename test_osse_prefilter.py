from cloudFilter import cloudFilter
import os

l1 = '/scratch/sao_atmos/ahsouri/Retrieval/methanesat_osses/L1/Global_no_airglow/'
#l1 = '/scratch/sao_atmos/ahsouri/Retrieval/methanesat_osses/L1/Global_no_airglow/MethaneSAT_20210830.0_02h_l2prof_l1.nc_subset'

msi = '/scratch/sao_atmos/jwilzews/DATA/MSI/MSI_clim'

fobj = cloudFilter(l1, msi, osse=True)
fobj.get_prefilter()
    
