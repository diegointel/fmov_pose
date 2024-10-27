# CUDA_VISIBLE_DEVICES=3 nohup ./scripts/barf_ho3d_AP13.bash > /dev/null &
# ours
python exp_runner.py --mode train --conf  ./confs/ho3d_barf.conf --case AP13