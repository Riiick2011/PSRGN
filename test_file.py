import os
cmd = 'python com_inference.py ./configs/thumos_i3d.yaml ./models/best_model/table1/CSSPGN_0.7048_0.33_model_GCN.pth'+' --sigma 0.33'#+str(sigma)
os.system(cmd)