# CUDA_VISIBLE_DEVICES=3 nohup ./scripts/ours_ml_spaceinvader.bash > /dev/null &

# ours
python exp_runner.py --mode train --conf  ./confs/ml_virtual.conf --case spaceinvader_ori --global_conf ./confs/ml_global_womask.conf