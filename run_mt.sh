# /bin/csh
# ----------------Parameters---------------------- #
#$ -S /bin/csh
#$ -pe mthread 4
#$ -q mThC.q
#$ -l mres=16G,h_data=4G,h_vmem=4G
#$ -cwd
#$ -j y
#$ -N prefilter
#$ -o out_mt.log
#
# ----------------Your Commands------------------- #
#python test_osse.py
python test_l1a.py
