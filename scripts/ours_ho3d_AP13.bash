# CUDA_VISIBLE_DEVICES=0 nohup ./scripts/ours_ho3d_AP13.bash > /dev/null &
# ours
python exp_runner.py --mode train --conf  ./confs/ho3d_virtual.conf --case AP13_ori --global_conf ./confs/ho3d_global_womask.conf