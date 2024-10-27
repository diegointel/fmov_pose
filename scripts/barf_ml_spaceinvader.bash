# CUDA_VISIBLE_DEVICES=2 nohup ./scripts/barf_ml_spaceinvader.bash > /dev/null &
# ours
python exp_runner.py --mode train --conf  ./confs/ml_barf.conf --case spaceinvader