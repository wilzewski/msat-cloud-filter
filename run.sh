# /bin/csh
# ----------------Parameters---------------------- #
#$ -S /bin/csh
#$ -q lThC.q
#$ -cwd
#$ -j y
#$ -N cloud
#$ -o out.log
#
# ----------------Your Commands------------------- #
python test_osse.py

