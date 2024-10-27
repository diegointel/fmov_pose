#CUDA_VISIBLE_DEVICES=2 nohup ./scripts/ours_ml_maneki-neko.bash > /dev/null &
# ours
python exp_runner.py --mode train --conf  ./confs/ml_virtual.conf --case maneki-neko_ori --global_conf ./confs/ml_global_womask.conf