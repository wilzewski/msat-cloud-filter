from cloudFilter import cloudFilter


l1 = '/scratch/sao_atmos/jwilzews/DATA/MAIR/L1B/RF06/CH4_NATIVE/MethaneAIR_L1B_CH4_20210806T163548_20210806T163557_20210930T114646.nc'

msi = '/scratch/sao_atmos/jwilzews/DATA/MSI/MSI_B11'

# L2 outputs
l2_ch4 = '/scratch/sao_atmos/jwilzews/DATA/MAIR/L2/RF06/CO2_Proxy_NATIVE/MethaneAIR_L1B_CH4_20210806T163548_20210806T163557_20210930T114646_CO2proxy.nc'
l2_o2 = '/scratch/sao_atmos/jwilzews/DATA/MAIR/L2/RF06/O2_NATIVE/MethaneAIR_L1B_O2_20210806T163548_20210806T163558_20210925T201103_CO2proxy.nc'
l2_h2o = '/scratch/sao_atmos/jwilzews/DATA/MAIR/L2/RF06/H2O_NATIVE/MethaneAIR_L1B_O2_20210806T163548_20210806T163558_20210925T201103_CO2proxy.nc'

fobj = cloudFilter(l1, msi, l2_ch4=l2_ch4, l2_o2=l2_o2, l2_h2o=l2_h2o)

fobj.get_filter()
fobj.apply_filter()
