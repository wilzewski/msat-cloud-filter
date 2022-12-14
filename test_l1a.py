from cloudFilter import cloudFilter

#l1 = '/scratch/sao_atmos/jwilzews/DATA/MAIR/L1A/Avionics_Only_MethaneAIR_L1B_CH4_20210811T162853_20210811T162923_20220505T094719.nc' # clear, mountains, lake, bright pixels
l1 = '/scratch/sao_atmos/jwilzews/DATA/MAIR/L1A/Avionics_Only_MethaneAIR_L1A_CH4_20210803T210345_20210803T210415_20221018T104602.nc'  # cloud

msi = '/scratch/sao_atmos/jwilzews/DATA/MSI/MSI_B11'

fobj = cloudFilter(l1, msi)

fobj.get_prefilter()

fobj.apply_prefilter()
