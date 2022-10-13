from cloudFilter import cloudFilter

l1 = '/scratch/sao_atmos/jwilzews/DATA/MAIR/L1A/Avionics_Only_MethaneAIR_L1B_O2_20210811T162853_20210811T162923_20220430T112041.nc'

msi = '/scratch/sao_atmos/jwilzews/DATA/MSI/MSI_B11'

fobj = cloudFilter(l1, msi)

fobj.get_prefilter()

fobj.apply_prefilter()
