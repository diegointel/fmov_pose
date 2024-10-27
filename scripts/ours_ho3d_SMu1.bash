# CUDA_VISIBLE_DEVICES=1 nohup ./scripts/ours_ho3d_SMu1.bash > /dev/null &
# ours
python exp_runner.py --mode train --conf  ./confs/ho3d_virtual.conf --case SMu1_ori --global_conf ./confs/ho3d_global_womask.conf