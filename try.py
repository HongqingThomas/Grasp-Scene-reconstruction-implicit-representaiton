# import torch
# print(torch.__version__) # 1.7; cudaversion10.2; python version3.8.0

import gzip
import pickle
import numpy as np
# with open('./GIGA/result/round_000/00000408.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data.values())

a = np.load('./GIGA/data/experiments/23-09-15-21-21-39/scenes/0afe2ac2e8e746868d54f685e76b63dc.npz')

print(a)
a.show()


# python scripts/sim_grasp_multiple.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best --vis --model ./data/models/giga_packed.pt --type giga --result-path ./result



# command

# 1. python scripts/sim_grasp_multiple.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best --vis --model ./data/models/giga_packed.pt --type giga --result-path result
# -  python scripts/sim_grasp_multiple.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best --vis --model ./data/models/giga_hyper_packed.pt --type giga_hyper --result-path result

# 2. python scripts/save_occ_data_parallel.py ./data/experiments/23-09-18-16-45-30/  100000 2 --num-proc 40

# 3. python scripts/eval_giga_1.py --dataset ./data/experiments/23-09-18-16-45-30/ --dataset_raw ./data/experiments/23-09-18-16-45-30/ --load-path ./data/models/giga_packed.pt

# trian new model command

# 1. python scripts/save_occ_data_parallel.py ./data_packed_train_raw/  100000 2 --num-proc 40

# 2. python scripts/train_giga.py --dataset ./data_packed_train_processed_dex_noise/ --dataset_raw ./data_packed_train_raw/

# eval IOU:
# python scripts/eval_geometry_voxel.py --model-path ./data/models/giga_packed.pt --type giga --dataset ./data_packed_train_processed_dex_noise/ --dataset_raw ./data_packed_train_raw/ --ROI

# result:
# 1) MERF train result:
# Train accuracy: 0.8781 precision: 0.8517 recall: 0.9156 loss_all: 0.3873 loss_qual: 0.2800 loss_rot: 0.1666 loss_width: 8.6746 loss_occ: 0.0702
# Val accuracy: 0.9026 precision: 0.8743 recall: 0.9402 loss_all: 0.3185 loss_qual: 0.2345 loss_rot: 0.1527 loss_width: 8.8984 loss_occ: 0.0534

# 2) MERF grasp result(no incremental training pipeline):
# Average planning time: 0.2905377394358317, total time: 0.29354905954996746
# Average results:
# Grasp sucess rate: 70.42 ± 2.05 %
# Declutter rate: 77.84 ± 3.18 %

# 3) incremental training pipeline:        
# self.sampling_number = 200 # if only ITP, the default should be 200, for MERF-ITP, fine-tune due to cuda out of memory
# self.N_samples = 0 #8
# self.N_surface = 16
# self.points_batch_size = 8
# Average planning time: 0.06861470712197794, total time: 0.07167491719529435
# Average results:
# Grasp sucess rate: 86.39 ± 1.81 %
# Declutter rate: 86.75 ± 1.86 %

# 4) incremental training pipeline:   
# self.sampling_number = 100 # if only ITP, the default should be 200, for MERF-ITP, fine-tune due to cuda out of memory
# self.N_samples = 0 #8
# self.N_surface = 16
# self.points_batch_size = 8 # if only ITP, the default should be 8, for MERF-ITP, fine-tune due to cuda out of memory
# self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
# self.incremental_iter_nums = 3 # 0
# Average planning time: 0.07021634015859912, total time: 0.07328695707042907
# Average results:
# Grasp sucess rate: 83.57 ± 2.15 %
# Declutter rate: 86.07 ± 1.95 %

# 5) eval the mesh:
# Geometry prediction results:
# iou: 0.787145
# chamfer-L1: 0.017904
# normals accuracy: 0.513864
# f-score: 0.245672
# iou_ROI: 0.830386
# iou_ROI_infer: 0.830696
# precision_ROI: 0.930622
# precision_ROI_infer: 0.928940
# recall_ROI: 0.884391
# recall_ROI_infer: 0.886302

# 512^2 * 3 1187 M
# 512^2 * 3 + ITP 2511 M 
# 512^2 * 3 + 64^3 2109 M
# 512^2 * 3 + 64^3 + ITP 4709 M
# 128^2 * 3 + 64^3 M 1663 M
# 128^2 * 3 + 64^3 + ITP  5G
# 256, 16  ITP 5956M
# 256, 16  1035 M