import math
import numpy as np


def Large_ravel(index, dim):
    ravel_index = 0
    multiplier = 1
    for i in reversed(range(len(dim))):
        ravel_index += index[i] * multiplier
        multiplier *= dim[i]
    return ravel_index


def Large_unravel(index, dim):
    unravel_index = [0] * len(dim)
    strides = [1] * len(dim)
    
    # compute strides
    for i in reversed(range(len(dim)-1)):
        strides[i] = strides[i+1] * dim[i+1]

    # unravel
    for i in range(len(dim)):
        unravel_index[i] = index // strides[i]
        index %= strides[i]

    return tuple(unravel_index)



def set_conversion(FI,FJ,dim):
    I = [[] for _ in  range(len(FI))]
    J = [[] for _ in range(len(FJ))]

    for i in range(len(FI)):
        if i==0:
            I[0] = [x for y in FI[0] for x in y]

        else:
            for j in FI[i]:
                for k in range(len(FI[i-1])):
                    if FI[i-1][k] == j[:i]:
                        I[i].append(Large_ravel([k,j[-1]],[len(FI[i-1]),dim[i]]))

    for i in range(len(FJ)-1,-1,-1):
        
        if i==len(FJ)-1:
            J[i] = [x for y in FJ[i] for x in y]
        
        else:
            for j in FJ[i]:
                for k in range(len(FJ[i+1])):
                    if FJ[i+1][k]==j[1:]:
                        J[i].append(Large_ravel([k,j[0]],[len(FJ[i+1]),dim[i+1]]))
                        break


    return I,J

def torch_ind_extract(info):
    FI_row = [[[int(info['lsets'][1][i][1])] for i in range(len(info['lsets'][1]))]]
    for j in range(2,len(info['lsets'])):
        FI_row.append([[int(ii) for ii in info['lsets'][j][i][1:]] for i in range(len(info['lsets'][j]))])

    FI_col = []
    for j in range(len(info['rsets'])-1):
        FI_col.append([[int(ii) for ii in info['rsets'][j][i][:len(info['rsets'][j][i])-1]] for i in range(len(info['rsets'][j]))])
    #FI_col.append([[int(info['rsets'][-2][i][0])] for i in range(len(info['rsets'][-2]))])
    

    return FI_row,FI_col
