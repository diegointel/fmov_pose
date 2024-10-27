# CUDA_VISIBLE_DEVICES=3 nohup ./scripts/gt_ho3d_AP13.bash > /dev/null &
# ours
python ./utils/official_neus_exp_runner.py --mode train --conf  ./confs/ho3d_gt.conf --case AP13