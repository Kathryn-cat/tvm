unbuffer python3 test_tensorization.py --work_dir ~/logs/matmul_run_1_orig --rpc_host 172.31.61.68 | tee ~/logs/matmul_run_1_orig.txt
unbuffer python3 test_tensorization.py --work_dir ~/logs/matmul_run_1_ten --use_tensorization --rpc_host 172.31.61.68 | tee ~/logs/matmul_run_1_ten.txt
