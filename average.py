import copy
import torch
from collections import defaultdict
from torch import nn

# Old averaging code is slightly better

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    assert(total_sample_num > 0)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        if ("num_batches_tracked" in k):
            continue
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg

# def average_weights2(w, s_num):
#     total_sample_num = sum(s_num)
#     assert (total_sample_num > 0)
#     w_avg = defaultdict(lambda: 0)
#
#     for i in range(0, len(w)):
#         for k in w[0].keys():
#             if ("num_batches_tracked" in k):
#                 continue
#             w_avg[k] += torch.mul(w[i][k], s_num[i]/total_sample_num)
#     return dict(w_avg)
