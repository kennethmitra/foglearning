import copy
import torch

def average_weights(w, s_num):
    """
    Average sets of model weights
    :param w: List of weights (List of dicts)
    :param s_num: Number of samples used to train each model (how much to weight each set of model weights)
    :return: Averaged weights
    """
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
