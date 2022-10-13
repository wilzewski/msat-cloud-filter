# /bin/csh
# ----------------Parameters---------------------- #
#$ -S /bin/csh
#$ -q uThC.q -l lopri
#$ -cwd
#$ -j y
#$ -N msi_clim
#$ -o out.log
#
# ----------------Your Commands------------------- #
#python test_osse.py
python test_l1a.py
