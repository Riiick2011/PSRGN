import torch
import torch.nn.functional as F

a=torch.tensor([20,20,20,4,4,7,7,20])
b=(a==4)*7+(a!=4)*20

gt_target = F.one_hot(
            a, num_classes=21
        )[:, :-1]


new_gt_target=F.one_hot(
            b, num_classes=21
        )[:, :-1]
g=gt_target+new_gt_target
print(g)