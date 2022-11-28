from cloudFilter import cloudFilter

l1 = '/scratch/sao_atmos/ahsouri/Retrieval/methanesat_osses/L1/Global_no_airglow/'

msi = '/scratch/sao_atmos/jwilzews/DATA/MSI/MSI_clim'

fobj = cloudFilter(l1, msi, osse=True)

fobj.get_filter()
