mode=$1

nohup python test.py grip apolloscape 0 ${mode} > log/grip_apolloscape_${mode}.log &
# nohup python test.py grip ngsim 0 > log/grip_ngsim.log &
# nohup python test.py grip nuscenes 0 > log/grip_nuscenes.log &

