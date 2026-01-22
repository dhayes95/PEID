import math
import numpy as np
import scipy as sp
import indexingfunctions
import random

def cheap_sample(low,high,unavailable,sample_size):
    
    num_samples = min([sample_size,(high - low)-len(unavailable)])
    samples = []
    selected = []
    for _ in range(num_samples):
        selected = list(unavailable)+samples
        avail = high - low - len(selected)
        b = random.randint(0,avail-1)
        
        srt_selected = sorted(selected)
        for i in range(len(selected)):
            if b - srt_selected[i]>=0:
                b+=1
        samples.append(b+low)
        

    return samples

def q_deim(V,cutoff = False):
   
    q,r,p = sp.linalg.qr(V.T,pivoting = True)
   
    if cutoff:
        p = p[:cutoff]
   
    return p

def sample_nested(row,col,sample_row,sample_col,tensor_dim,axis = 0):

    FI= [[] for _ in range(len(row))]
    FJ = [[] for _ in range(len(col))]
    if axis == 0:
        for i in range(len(tensor_dim)-1):
            FJ[i] = list(col[i])
            unavail = [indexingfunctions.Large_ravel(j,tensor_dim[i+1:]) for j in col[i]]
            selected = cheap_sample(0,math.prod(tensor_dim[i+1:]),unavail,sample_col[i])
            for j in selected:
                FJ[i].append(list(indexingfunctions.Large_unravel(j,tensor_dim[i+1:])))


            if i==0:
                FI[i] = list(row[i])
                unavail = [row[i][j][-1] for j in range(len(row[i]))]
                selected = cheap_sample(0,tensor_dim[0],unavail,sample_row[i])
                for j in selected:
                    FI[i].append([j])
            else:
                FI[i] = list(row[i])
                unavail = []
                for j in row[i]:
                    count = 0
                    for k in range(len(row[i-1])):
                        if row[i-1][k]==j[:i]:
                            unavail.append(indexingfunctions.Large_ravel([k,j[-1]],[len(FI[i-1]),tensor_dim[i]]))
                            break
                        else:
                            k+=1            
                locs = cheap_sample(0,len(FI[i-1])*tensor_dim[i],unavail,sample_row[i])
                unrav = [indexingfunctions.Large_unravel(j,[len(FI[i-1]),tensor_dim[i]]) for j in locs]
                for j in unrav:
                    FI[i].append(FI[i-1][j[0]]+[j[1]])
    else:
        for i in range(len(tensor_dim)-1):
            FI[i]  = list(row[i])
            unavail = [indexingfunctions.Large_ravel(j,tensor_dim[:i+1]) for j in row[i]]
            selected = cheap_sample(0,math.prod(tensor_dim[:i+1]),unavail,sample_row[i])
            for j in selected:
                FI[i].append(list(indexingfunctions.Large_unravel(j,tensor_dim[:i+1])))

        for i in range(len(col)-1,-1,-1):
            FJ[i] = list(col[i])

            if i==len(col)-1:
                unavail = [col[i][j][0] for j in range(len(col[i]))]
                selected = cheap_sample(0,tensor_dim[-1],unavail,sample_col[i])
                for j in selected:
                    FJ[i].append([j])
            else:
                unavail = []
                for j in col[i]:
                    for k in range(len(col[i+1])):
                        if col[i+1][k]==j[i+1:]:
                            unavail.append(indexingfunctions.Large_ravel([j[0],k],[tensor_dim[i+1],len(FJ[i+1])]))
                            break
                locs = cheap_sample(0,len(FJ[i+1])*tensor_dim[i+1],unavail,sample_col[i])
                unrav = [indexingfunctions.Large_unravel(j,[tensor_dim[i+1],len(FJ[i+1])]) for j in locs]
                for j in unrav:
                    FJ[i].append([j[0]] + FJ[i+1][j[1]])

    return FI,FJ


def tuple_sample(low,high,unavail,sample_size,dim):

    high = [int(i) for i in high]
    high_end = indexingfunctions.Large_ravel(high,dim)
    low_end = indexingfunctions.Large_ravel(low,dim)
    rav_unavail = [indexingfunctions.Large_ravel(i,dim) for i in unavail]
    samples = cheap_sample(low_end, high_end,rav_unavail, sample_size)
    sample_tuples = [list(indexingfunctions.Large_unravel(i,dim)) for i in samples]
    
    return sample_tuples




