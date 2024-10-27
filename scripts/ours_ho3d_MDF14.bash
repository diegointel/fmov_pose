#CUDA_VISIBLE_DEVICES=1 nohup ./scripts/ours_ho3d_MDF14.bash > /dev/null &
# ours
python exp_runner.py --mode train --conf  ./confs/ho3d_virtual.conf --case MDF14_ori --global_conf ./confs/ho3d_global_womask.conf