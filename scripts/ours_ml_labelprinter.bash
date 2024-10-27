# CUDA_VISIBLE_DEVICES=0 nohup ./scripts/ours_ml_labelprinter.bash > /dev/null &
# ours
python exp_runner.py --mode train --conf  ./confs/ml_virtual.conf --case labelprinter_ori --global_conf ./confs/ml_global_womask.conf