import numpy as np
import scipy as sp
import math
import random
try:
    from mpi4py import MPI
except Exception:
    MPI = None


# Local helpers keep this module independent of other project files.

def large_ravel(index, dim):
    ravel_index = 0
    multiplier = 1
    for i in reversed(range(len(dim))):
        ravel_index += index[i] * multiplier
        multiplier *= dim[i]
    return ravel_index


def large_unravel(index, dim):
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
                        I[i].append(large_ravel([k,j[-1]],[len(FI[i-1]),dim[i]]))

    for i in range(len(FJ)-1,-1,-1):
        
        if i==len(FJ)-1:
            J[i] = [x for y in FJ[i] for x in y]
        
        else:
            for j in FJ[i]:
                for k in range(len(FJ[i+1])):
                    if FJ[i+1][k]==j[1:]:
                        J[i].append(large_ravel([k,j[0]],[len(FJ[i+1]),dim[i+1]]))
                        break


    return I,J


def torch_ind_extract(info):
    """Convert TnTorch cross index sets to PEID row and column index lists."""
    FI_row = [[[int(info['lsets'][1][i][1])]
               for i in range(len(info['lsets'][1]))]]
    for j in range(2, len(info['lsets'])):
        FI_row.append([
            [int(ii) for ii in info['lsets'][j][i][1:]]
            for i in range(len(info['lsets'][j]))
        ])

    FI_col = []
    for j in range(len(info['rsets']) - 1):
        FI_col.append([
            [int(ii) for ii in info['rsets'][j][i][:-1]]
            for i in range(len(info['rsets'][j]))
        ])

    return FI_row, FI_col


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


def sample_nested(row,col,sample_row,sample_col,tensor_dim,axis = 0):

    FI= [[] for _ in range(len(row))]
    FJ = [[] for _ in range(len(col))]
    if axis == 0:
        for i in range(len(tensor_dim)-1):
            FJ[i] = list(col[i])
            unavail = [large_ravel(j,tensor_dim[i+1:]) for j in col[i]]
            selected = cheap_sample(0,math.prod(tensor_dim[i+1:]),unavail,sample_col[i])
            for j in selected:
                FJ[i].append(list(large_unravel(j,tensor_dim[i+1:])))


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
                            unavail.append(large_ravel([k,j[-1]],[len(FI[i-1]),tensor_dim[i]]))
                            break
                        else:
                            k+=1            
                locs = cheap_sample(0,len(FI[i-1])*tensor_dim[i],unavail,sample_row[i])
                unrav = [large_unravel(j,[len(FI[i-1]),tensor_dim[i]]) for j in locs]
                for j in unrav:
                    FI[i].append(FI[i-1][j[0]]+[j[1]])
    else:
        for i in range(len(tensor_dim)-1):
            FI[i]  = list(row[i])
            unavail = [large_ravel(j,tensor_dim[:i+1]) for j in row[i]]
            selected = cheap_sample(0,math.prod(tensor_dim[:i+1]),unavail,sample_row[i])
            for j in selected:
                FI[i].append(list(large_unravel(j,tensor_dim[:i+1])))

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
                            unavail.append(large_ravel([j[0],k],[tensor_dim[i+1],len(FJ[i+1])]))
                            break
                locs = cheap_sample(0,len(FJ[i+1])*tensor_dim[i+1],unavail,sample_col[i])
                unrav = [large_unravel(j,[tensor_dim[i+1],len(FJ[i+1])]) for j in locs]
                for j in unrav:
                    FJ[i].append([j[0]] + FJ[i+1][j[1]])

    return FI,FJ


def tuple_sample(low,high,unavail,sample_size,dim):

    high = [int(i) for i in high]
    high_end = large_ravel(high,dim)
    low_end = large_ravel(low,dim)
    rav_unavail = [large_ravel(i,dim) for i in unavail]
    samples = cheap_sample(low_end, high_end,rav_unavail, sample_size)
    sample_tuples = [list(large_unravel(i,dim)) for i in samples]
    
    return sample_tuples


def unfolding_submatrix(tensor_entry,uFI,uFJ,k,tensor_dim,axis,full = True):
    if axis == 0:
        if full:
            if k==0:
            
                fst_indt = np.repeat(uFI[0],tensor_dim[k+1]*len(uFJ[k+1]))
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1])*len(uFI[k]))
                lst_indt = np.tile(np.repeat(uFJ[k+1],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]
                
                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[0]),tensor_dim[1]*len(uFJ[k+1])])
            elif k==len(tensor_dim)-2:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1],axis = 0)
                lst_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFI[k]))


                lists = [fst_indt,lst_indt]
                ind = np.column_stack(lists)

                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]])
                
            else:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1]*len(uFJ[k+1]),axis = 0)
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1])*len(uFI[k]))
                lst_indt = np.tile(np.repeat(uFJ[k+1],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]


                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]*len(uFJ[k+1])])

        else:
            
            if k==len(tensor_dim)-2:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1],axis = 0)
                lst_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFI[k]))


                lists = [fst_indt,lst_indt]
                ind = np.column_stack(lists)

                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]])
                
            else:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1],axis = 0)
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFI[k]))
                lst_indt = np.tile(np.repeat([uFJ[k+1][-1]],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]


                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]])

    elif axis == 1:
        if full:
            if k==0:
                
                fst_ind = np.repeat(np.arange(tensor_dim[k]),len(uFJ[k]),axis = 0)
                lst_ind = np.tile(uFJ[k],(tensor_dim[k],1))
                lists = [fst_ind,lst_ind]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[tensor_dim[k],len(uFJ[k])])
                
            elif k == len(tensor_dim)-2:

                fst_ind = np.repeat(uFI[k-1],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k])),len(uFI[k-1]))
                lst_ind = np.tile(uFJ[k],(len(uFI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1])*tensor_dim[k],len(uFJ[k])])

            else:

                fst_ind = np.repeat(uFI[k-1],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k])),len(uFI[k-1]))
                lst_ind = np.tile(uFJ[k],(len(uFI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1])*tensor_dim[k],len(uFJ[k])]) 
        else:
            if k==0:
                
                fst_ind = np.repeat(np.arange(tensor_dim[k]),len(uFJ[k]),axis = 0)
                lst_ind = np.tile(uFJ[k],(tensor_dim[k],1))
                lists = [fst_ind,lst_ind]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[tensor_dim[k],len(uFJ[k])])

            else:

                fst_ind = np.repeat([uFI[k-1][-1]],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.repeat(np.arange(tensor_dim[k]),len(uFJ[k]))
                lst_ind = np.tile(uFJ[k],(tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]
                
                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[tensor_dim[k],len(uFJ[k])]) 
    return submatrix


def tt_eval(cores,index):
    index_array = np.asarray(index, dtype=int)

    if index_array.ndim == 1:
        start = cores[0][:,index_array[0],:]
        for i in range(1,len(cores)):
            start = start@cores[i][:,index_array[i],:]
        return start[0][0]

    if index_array.ndim != 2:
        raise ValueError("index must be a 1D index or a 2D array of indices")

    if index_array.shape[1] != len(cores):
        raise ValueError("each index must have one entry per TT core")

    start = np.transpose(cores[0][:,index_array[:,0],:], (1,0,2))
    for i in range(1,len(cores)):
        next_core = np.transpose(cores[i][:,index_array[:,i],:], (1,0,2))
        start = np.einsum('bij,bjk->bik', start, next_core)

    return start[:,0,0]


def sampled_error(tensor_entry,dim,samples,*args,relative=True,seed=10):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, np.asarray(dim), size=(samples, len(dim)))
    true_values = np.asarray(tensor_entry(indices,dim))
    norm = np.sum(true_values**2)

    errors = []
    for cores in args:
        approx_values = np.asarray(tt_eval(cores,indices))
        error = np.sum((true_values - approx_values)**2)
        if relative:
            errors.append(np.sqrt(error) / np.sqrt(norm))
        else:
            errors.append(np.sqrt(error))
    return errors


def sampled_norm(dim,samples,*args,seed=10):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, np.asarray(dim), size=(samples, len(dim)))

    vals = []
    for cores in args:
        approx_values = np.asarray(tt_eval(cores,indices))
        val = np.sum(( approx_values)**2)
        vals.append(val)


    return [np.sqrt(math.prod(dim)*vals[i]/samples) for i in range(len(args))]


def tt_rounding(cores,eps):
    """Round a TT tensor using Algorithm 2 from Oseledets (2011)."""
    if len(cores) == 0:
        return []

    if len(cores) == 1:
        return [np.array(cores[0], copy=True)]

    rounded_cores = [np.array(core, copy=True) for core in cores]
    d = len(rounded_cores)

    # Compute the Frobenius norm directly from the TT cores.
    #gram = np.array([[1.0]])
    #for core in rounded_cores:
    #    gram = np.einsum('ab,aic,bid->cd', gram, core, core)
    #tt_norm0 = np.sqrt(np.abs(gram[0,0]))
    tt_norm = sampled_norm([i.shape[1] for i in cores],1000,cores)[0]
    #print(tt_norm0,tt_norm)
    delta = (eps * tt_norm) / np.sqrt(d - 1)

    # Right-to-left orthogonalization via QR with orthonormal rows.
    for k in range(d - 1, 0, -1):
        core = rounded_cores[k]
        r_prev, mode_size, r_next = core.shape
        matrix = np.reshape(core, (r_prev, mode_size * r_next))

        q_t, r_t = np.linalg.qr(matrix.T, mode='reduced')
        q_rows = q_t.T
        r_factor = r_t.T

        rounded_cores[k] = np.reshape(q_rows, (q_rows.shape[0], mode_size, r_next))
        rounded_cores[k - 1] = np.tensordot(rounded_cores[k - 1], r_factor, axes=([2], [0]))

    # Left-to-right compression with delta-truncated SVD.
    for k in range(d - 1):
        core = rounded_cores[k]
        r_prev, mode_size, r_next = core.shape
        matrix = np.reshape(core, (r_prev * mode_size, r_next))

        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        new_rank = max(1, np.count_nonzero(s > delta))

        rounded_cores[k] = np.reshape(u[:, :new_rank], (r_prev, mode_size, new_rank))
        transfer = np.diag(s[:new_rank]) @ vh[:new_rank, :]
        rounded_cores[k + 1] = np.tensordot(transfer, rounded_cores[k + 1], axes=([1], [0]))

    return rounded_cores


def tt_add(core1,core2,tensor_dim):
    trunc_ranks_1 = [core1[i].shape[0] for i in range(1,len(core1))]
    #trunc_ranks_2 = [core2[i].shape[0] for i in range(1,len(core2))]
    
    cores = []
    cores.append(np.concatenate((core1[0],core2[0]),axis = 2))
   
    for i in range(1,len(core1)-1):
        cores.append(np.zeros((core1[i].shape[0]+core2[i].shape[0],tensor_dim[i],core1[i].shape[2]+core2[i].shape[2])))
        
        cores[i][:trunc_ranks_1[i-1],:,:trunc_ranks_1[i]] = core1[i]
        cores[i][trunc_ranks_1[i-1]:,:,trunc_ranks_1[i]:] = core2[i]
   
       
    cores.append(np.concatenate((core1[-1],core2[-1]),axis = 0))
   
    return cores


def tt_round(cores,eps,rank):
    rounded_cores = [cores[0]]
    dim = [i.shape[1] for i in cores]
    
    #Right to left sweep through cores for orthogonalization


    for i in range(len(cores)-1):
        core_size = rounded_cores[i].shape
        #print(i)
        q,r = np.linalg.qr(np.reshape(rounded_cores[i],[math.prod(core_size[:2]),core_size[-1]]))
        u,s,v = np.linalg.svd(r)
        u = q@u
        #u,s,v = np.linalg.svd(np.reshape(rounded_cores[i],[math.prod(core_size[:2]),core_size[-1]]))
        if i==0:
            rounded_cores[i] = np.reshape(u[:,:rank[i]],[1,dim[i],rank[i]])
        else:
            rounded_cores[i] = np.reshape(u[:,:rank[i]],[rank[i-1],dim[i],rank[i]])
        
        R = np.diag(s[:rank[i]])@v[:rank[i],:]
        next_core_size = cores[i+1].shape
        G = np.reshape(cores[i+1],[next_core_size[0],math.prod(next_core_size[1:])])
        #print((R@G).shape)
        rounded_core_size = tuple([rank[i]])+next_core_size[1:]
        rounded_cores.append(np.reshape(R@G,rounded_core_size))
    
    
    return rounded_cores


def rank_cutoff(s_values, adap_tol):
    if len(s_values) == 0:
        return 1
    leading = s_values[0]
    if np.isclose(leading, 0.0):
        return 1
    norm_s = np.asarray(s_values) / leading
    return max(1, int(np.count_nonzero(norm_s >= adap_tol)))


def peid_par2(tensor_entry,FI_row,FI_col,sample_row,sample_col,tensor_dim,adap_tol = None,return_trunc_ranks=False):
   
    cores_left = []
    uj_left = []
   
    cores_right_rev = []
    uj_right = []
    trunc_ranks = [len(i) for i in FI_row]
   
    d = len(tensor_dim)
    n = len(trunc_ranks)
    mid_l = math.floor((d-1)/2)
    if not (d-1)%2:
        mid_r = math.floor((d-1)/2)
    else:
        mid_r = math.floor((d-1)/2)

    FI_left,FJ_left = sample_nested(FI_row, FI_col, sample_row, sample_col, tensor_dim,0)
    I_left,J_left = set_conversion(FI_left, FJ_left, tensor_dim)
    
    FI_right,FJ_right = sample_nested(FI_row, FI_col, sample_row, sample_col, tensor_dim,1)
    I_right,J_right = set_conversion(FI_right, FJ_right, tensor_dim)

    for i in range(n,mid_r,-1):
        if i == n:            
            mat = unfolding_submatrix(tensor_entry, FI_right, FJ_right, i-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i-1] = cutoff
            u = q@U[:,:trunc_ranks[i-1]]
            
            uj_right.append(u)
            cores_right_rev.append(np.reshape(uj_right[-1],[1,tensor_dim[i],trunc_ranks[i-1]]))
        elif i==0:
            mat = unfolding_submatrix(tensor_entry, FI_right, FJ_right, 0, tensor_dim, 1).T
            A = sp.linalg.lstsq(uj_right[-1][J_right[0],:],mat,lapack_driver="gelsy")[0]
            cores_right_rev.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],1]))
        else:
            
            mat = unfolding_submatrix(tensor_entry, FI_right, FJ_right, i-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i-1] = cutoff
            u = q@U[:,:trunc_ranks[i-1]]
            uj_right.append(u)
            uj_tilde = np.reshape(uj_right[-1],[len(FJ_right[i]),tensor_dim[i]*trunc_ranks[i-1]])  

            A = sp.linalg.lstsq(uj_right[-2][J_right[i],:],uj_tilde,lapack_driver="gelsy")[0]
            cores_right_rev.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],trunc_ranks[i-1]]))
	      
    cores_right = [np.transpose(cores_right_rev[i],[2,1,0]) for i in range(len(cores_right_rev)-1,-1,-1)]
   
    for i in range(mid_l):
        if i == n:
            mat = unfolding_submatrix(tensor_entry, FI_left, FJ_left, i-1, tensor_dim, 0)
            A = sp.linalg.lstsq(uj_left[-1][I_left[i-1],:],mat,lapack_driver="gelsy")[0]
            cores_left.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],1]))
        elif i==0:
            mat = unfolding_submatrix(tensor_entry, FI_left, FJ_left, i, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i] = cutoff
            u = q@U[:,:trunc_ranks[i]]
            uj_left.append(u)
            cores_left.append(np.reshape(uj_left[-1],[1,tensor_dim[i],trunc_ranks[i]]))
        else:
            mat = unfolding_submatrix(tensor_entry, FI_left, FJ_left, i, tensor_dim, 1)
            
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i] = cutoff
            u = q@U[:,:trunc_ranks[i]]
            uj_left.append(u)
            uj_tilde = np.reshape(uj_left[-1],[len(FI_left[i-1]),tensor_dim[i]*trunc_ranks[i]])
            A = sp.linalg.lstsq(uj_left[-2][I_left[i-1],:],uj_tilde,lapack_driver="gelsy")[0]
            cores_left.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
    
    fst_ind = np.repeat(FI_left[mid_l-1],tensor_dim[mid_l]*len(FJ_right[mid_l]),axis = 0)
    mid_ind = np.tile(np.repeat(np.arange(tensor_dim[mid_l]),len(FJ_right[mid_l])),len(FI_left[mid_l-1]))
    lst_ind = np.tile(FJ_right[mid_l],(len(FI_left[mid_l-1])*tensor_dim[mid_l],1))
    ind_t = np.column_stack([fst_ind,mid_ind,lst_ind])
    t = np.reshape(tensor_entry(ind_t,tensor_dim),[len(FI_left[mid_l-1]),tensor_dim[mid_l]*len(FJ_right[mid_l])])
    t = sp.linalg.lstsq(uj_left[-1][I_left[mid_l-1],:],t,lapack_driver="gelsy")[0]
    t = np.reshape(t,[trunc_ranks[mid_l-1]*tensor_dim[mid_l],len(J_right[mid_l])])

    r = sp.linalg.lstsq(uj_right[-1][J_right[mid_l],:],t.T,lapack_driver="gelsy")[0].T
    t1 = np.reshape(r,[trunc_ranks[mid_l-1],tensor_dim[mid_l],trunc_ranks[mid_l]])
   
    cores_all = cores_left
    cores_all.append(t1)
    for i in cores_right:
        cores_all.append(i)
   
    if return_trunc_ranks:
        return cores_all,trunc_ranks

    return cores_all


def peid_seq2(tensor_entry,FIt,FJt,sample_row,sample_col,tensor_dim,adap_tol = None,return_trunc_ranks=False):
   
    cores = []
    cores_left = []
   
    cores_right = []
    trunc_ranks = [len(i) for i in FIt]
   
    d = len(tensor_dim)
    n = len(trunc_ranks)
    mid_l = math.floor((d-1)/2)
    if not (d-1)%2:
        mid_r = math.floor((d-1)/2)
    else:
        mid_r = math.floor((d-1)/2)
        
    K_list = []
    L_list = []

    for i in range(len(tensor_dim)-1):
        L_list.append(tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
    for i in range(len(tensor_dim)-1):
        K_list.append(tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))
 
    FJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
    FI = [FIt[i] + K_list[i] for i in range(len(FIt))]
    
        
    for i in range(n,mid_r,-1):
        if i == len(tensor_dim)-1:
            mat = unfolding_submatrix(tensor_entry, FI,FJ, i-1, tensor_dim, 0).T
            Q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i-1] = cutoff
            q = Q@U[:,:trunc_ranks[i-1]]
            
            cores_right.append(np.reshape(q.T,[trunc_ranks[i-1],tensor_dim[i],1]))
        else:
            G = np.zeros((trunc_ranks[i],len(FJ[i])))
            for j in range(len(FJ[i])):
                ind = list(FJ[i][j])
                col = cores_right[0][:,ind[0],:]
                for l in range(1,len(cores_right)):
                    col = col@cores_right[l][:,ind[l],:]
                G[:,[j]] = col
            
            mat = unfolding_submatrix(tensor_entry, FI,FJ, i, tensor_dim, 1)
            y = sp.linalg.lstsq(G.T,mat.T,lapack_driver="gelsy")[0].T
            
            
            if i>0:
                y= np.reshape(y,[len(FI[i-1]),tensor_dim[i]*trunc_ranks[i]]).T
                Q,r = sp.linalg.qr(y,mode = "economic")
                U,s_values,_ = np.linalg.svd(r)
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    trunc_ranks[i-1] = cutoff
                q = Q@U[:,:trunc_ranks[i-1]]
                
                cores_right.insert(0,np.reshape(q.T,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
            else:
                cores_right.insert(0,np.reshape(y,[1,tensor_dim[0],trunc_ranks[0]]))
    
    for i in range(mid_l):
        if i==0:
            B_test = unfolding_submatrix(tensor_entry, FI,FJ, i, tensor_dim, 1)
            Q,r = sp.linalg.qr(B_test,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i] = cutoff
            q = Q@U[:,:trunc_ranks[i]]
            
            cores_left.append(np.reshape(q,[1,tensor_dim[i],trunc_ranks[i]]))
        elif i==len(tensor_dim)-1:
            ind_row = [large_unravel(ir,tensor_dim[:i]) for ir in list(tuple(FI[i-1]))]

            A_test = unfolding_submatrix(tensor_entry, FI,FJ, i-1, tensor_dim, 0)
            F = []
            for ir in ind_row:
                row  = cores_left[0][:,ir[0],:]
                for j in range(1,i):
                    row = row@cores_left[j][:,ir[j],:]
                F.append(row[0])
            F = np.array(F)
            B = np.linalg.lstsq(F,A_test,rcond = None)[0]
            cores_left.append(np.reshape(B,[trunc_ranks[i-1],tensor_dim[i],1]))
        else:
            ind_row = list(FI[i-1])
            fst_ind = np.repeat(FI[i-1],tensor_dim[i]*len(FJ[i]),axis = 0)
            mid_ind = np.tile(np.repeat(np.arange(tensor_dim[i]),len(FJ[i])),len(FI[i-1]))
            lst_ind = np.tile(FJ[i],(len(FI[i-1])*tensor_dim[i],1))
            full_inds = np.column_stack([fst_ind,mid_ind,lst_ind])
            
            t = np.reshape(tensor_entry(full_inds,tensor_dim),[len(FI[i-1]),tensor_dim[i]*len(FJ[i])])
            F = []
            for ir in ind_row:
                row  = cores_left[0][:,ir[0],:]
                for j in range(1,i):
                    row = row@cores_left[j][:,ir[j],:]
                F.append(row[0])
            F = np.array(F)

            B = np.reshape(sp.linalg.lstsq(F,t,lapack_driver="gelsy")[0],[trunc_ranks[i-1]*tensor_dim[i],len(FJ[i])])

            Q,r = sp.linalg.qr(B,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i] = cutoff
            q = Q@U[:,:trunc_ranks[i]]
            cores_left.append(np.reshape(q,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
   
    i=mid_l
    ind_row = list(FI[i-1])

    F = []
    for ir in ind_row:
        row  = cores_left[0][:,ir[0],:]
        for j in range(1,i):
            row = row@cores_left[j][:,ir[j],:]
        F.append(row[0])
    F = np.array(F)
    
    G = np.zeros((trunc_ranks[i],len(FJ[i])))
    for j in range(len(FJ[i])):
        ind = list(FJ[i][j])
        col = cores_right[0][:,ind[0],:]
        for l in range(1,len(cores_right)):
            col = col@cores_right[l][:,ind[l],:]
        G[:,[j]] = col
    
    fst_ind = np.repeat(FI[mid_l-1],tensor_dim[mid_l]*len(FJ[mid_l]),axis = 0)
    mid_ind = np.tile(np.repeat(np.arange(tensor_dim[mid_l]),len(FJ[mid_l])),len(FI[mid_l-1]))
    lst_ind = np.tile(FJ[mid_l],(len(FI[mid_l-1])*tensor_dim[mid_l],1))
    ind_t = np.column_stack([fst_ind,mid_ind,lst_ind])
    
    t = np.reshape(tensor_entry(ind_t,tensor_dim),[len(FI[i-1]),tensor_dim[i]*len(FJ[i])])
    
    inter = np.linalg.lstsq(F,t,rcond = None)[0]
    inter = np.reshape(inter,[tensor_dim[mid_l]*trunc_ranks[mid_l-1],len(FJ[mid_l])])
    
    y = np.linalg.lstsq(G.T,inter.T,rcond = None)[0].T
    cores_left.append(np.reshape(y,[trunc_ranks[mid_l-1],tensor_dim[mid_l],trunc_ranks[mid_l]]))
    
    cores = cores_left + cores_right
    if return_trunc_ranks:
        return cores,trunc_ranks

    return cores


def peid_par(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = None, return_trunc_ranks=False, orientation = "ltr"):

    if orientation == "ltr":

        trunc_ranks = [len(i) for i in FI]
        cores = []
        uj = []
        uj_tilde = []

        n = len(trunc_ranks)

        FI = [[[int(i) for i in FI[j][k]] for k in range(len(FI[j]))] for j in range(len(FI))]
        FJ = [[[int(i) for i in FJ[j][k]] for k in range(len(FJ[j]))] for j in range(len(FJ))]

        uFI,uFJ = sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,0)
        uK,uL = set_conversion(uFI, uFJ, tensor_dim)

        I = uK
        J = uL

        uFI = [np.array(i) for i in uFI]
        uFJ = [np.array(i) for i in uFJ]

        for i in range(n+1):
            if i == n:
                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0)
                #print(f"Condition number for mode {i-1}: {np.linalg.cond(uj[-1][I[i-1],:])}")
                A = sp.linalg.lstsq(uj[-1][I[i-1],:],mat,cond = None)[0]
                cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],1]))
            elif i == 0:
                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
                q,r = sp.linalg.qr(mat,mode = "economic")
                U,s_values,_ = sp.linalg.svd(r)
            
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    trunc_ranks[i] = cutoff
                u = q@U[:,:trunc_ranks[i]]
                uj.append(u)
                cores.append(np.reshape(uj[0],[1,tensor_dim[i],trunc_ranks[i]]))
            else:
                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
                q,r = sp.linalg.qr(mat,mode = "economic")
                U,s_values,_ = sp.linalg.svd(r)
            
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    trunc_ranks[i] = cutoff
                u = q@U[:,:trunc_ranks[i]]
                uj.append(u)
                #print(f"Condition number for mode {i}: {np.linalg.cond(uj[i-1][I[i-1],:])}")
                uj_tilde = np.reshape(uj[i],[len(uFI[i-1]),tensor_dim[i]*trunc_ranks[i]])
                A = sp.linalg.lstsq(uj[i-1][I[i-1],:],uj_tilde,cond = None)[0]
                cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))

        if return_trunc_ranks:
            return cores, trunc_ranks

        return cores

    elif orientation == "rtl":


        trunc_ranks = [len(i) for i in FI]
        cores = []
        uj = []
        uj_tilde = []

        cores_adap = []
        uj_adap = []
        uj_tilde_adap = []
        trunc_ranks_adap = []

        n = len(trunc_ranks)
        superblock_time = 0

        uFI,uFJ = sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,1)
        uK,uL = set_conversion(uFI,uFJ, tensor_dim)
        I = uK
        J = uL

        uFI = [np.array(i) for i in uFI]
        uFJ = [np.array(i) for i in uFJ]

        for i in range(n,-1,-1):
            if i == n:  
                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
                q,r = sp.linalg.qr(mat,mode = "economic")
                U,s_values,_ = np.linalg.svd(r)

                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    trunc_ranks[i-1] = cutoff
            
                u = q@U
                uj.append(u[:,:trunc_ranks[i-1]])
                cores.append(np.reshape(uj[-1],[1,tensor_dim[i],trunc_ranks[i-1]]))

            elif i==0:

                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, 0, tensor_dim, 1).T
                A = sp.linalg.lstsq(uj[-1][J[0],:],mat,lapack_driver="gelsy")[0]
                cores.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],1]))

            else:
            
                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
                q,r = sp.linalg.qr(mat,mode = "economic")
                U,s_values,_ = np.linalg.svd(r)
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    trunc_ranks[i-1] = cutoff
            
                u = q@U
            
                uj.append(u[:,:trunc_ranks[i-1]])

                uj_tilde = np.reshape(uj[-1],[len(uFJ[i]),tensor_dim[i]*trunc_ranks[i-1]])

                A = sp.linalg.lstsq(uj[-2][J[i],:],uj_tilde,lapack_driver="gelsy")[0]
                cores.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],trunc_ranks[i-1]]))


        core_list = [np.transpose(cores[i],[2,1,0]) for i in range(len(tensor_dim)-1,-1,-1)]
        if return_trunc_ranks:
            return core_list, trunc_ranks

        return core_list

    else:
        raise ValueError("orientation must be ltr or rtl")


def peid_r(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,return_trunc_ranks = False,base = "par"):

    trunc_ranks = [len(i) for i in FI]

    if base == "par":
        cores_ltr = peid_par(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,orientation = "ltr")
        cores_rtl = peid_par(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,orientation = "rtl")
    elif base == "seq":
        cores_ltr = peid_seq(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,orientation = "ltr")
        cores_rtl = peid_seq(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,orientation = "rtl")
    else:
        raise ValueError("base must be par or seq")

    cores = tt_add(cores_ltr,cores_rtl,tensor_dim)
    cores[0] *= 0.5
    cores = tt_round(cores,0,trunc_ranks)

    if return_trunc_ranks:
        return cores, trunc_ranks

    return cores


def ttaca_cores(tensor_entry,FI,FJ,trunc_ranks,tensor_dim,X_dict):

    cores = []
    uj = []
    uj_tilde = []
    n = len(trunc_ranks)

    uFI = [[[int(i) for i in FI[j][k]] for k in range(len(FI[j]))] for j in range(len(FI))]
    uFJ = [[[int(i) for i in FJ[j][k]] for k in range(len(FJ[j]))] for j in range(len(FJ))]

    #uFI,uFJ = sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,0)
    uK,uL = set_conversion(uFI, uFJ, tensor_dim)


    I = uK
    J = uL

    uFI = [np.array(i) for i in uFI]
    uFJ = [np.array(i) for i in uFJ]


    for i in range(n+1):
        if i == n:
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0)            
            
            A = np.linalg.lstsq(uj[-1][I[i-1],:],mat,rcond = None)[0]
            cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],1]))
        elif i==0:
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            #q,r = sp.linalg.qr(mat,mode = "economic")
            #U,s_values,_ = np.linalg.svd(r)
            #u = q@U
            uj.append(mat)
            cores.append(np.reshape(uj[0],[1,tensor_dim[i],trunc_ranks[i]]))
        else:
           
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            #q,r = sp.linalg.qr(mat,mode = "economic")
            #U,s_values,_ = np.linalg.svd(r)
            #u = q@U
            uj.append(mat)

            
            uj_tilde = np.reshape(uj[i],[len(uFI[i-1]),tensor_dim[i]*trunc_ranks[i]])

            
            A = np.linalg.lstsq(uj[i-1][I[i-1],:],uj_tilde,rcond = None)[0]

            cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
    
    return cores


def peid_seq(tensor_entry,FIt,FJt,sample_row,sample_col,tensor_dim,adap_tol = None,return_trunc_ranks = False,orientation = "ltr"):

    if orientation == "ltr":
   
        truncation_rank = [len(i) for i in FIt]

        cores = []
        cores_adap = []
        trunc_ranks_adap = []

        K_list = []
        L_list = []
    
    
        for i in range(len(tensor_dim)-1):
            L_list.append(tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
        for i in range(len(tensor_dim)-1):
            K_list.append(tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))

        uFJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
        uFI = [FIt[i] + K_list[i] for i in range(len(FIt))]
    
        for i in range(len(tensor_dim)):
            if i==0:
            
                B_test = unfolding_submatrix(tensor_entry,uFI,uFJ, i, tensor_dim, 1)

                q,r = sp.linalg.qr(B_test,mode = "economic")
                U,s_values,_ = np.linalg.svd(r)
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    truncation_rank[i] = cutoff
                u = q@U
                #fixed_rank = _bounded_rank(truncation_rank[i], u.shape[1])
                cores.append(np.reshape(u[:,:truncation_rank[i]],[1,tensor_dim[i],truncation_rank[i]]))

            elif i==len(tensor_dim)-1:

                A_test = unfolding_submatrix(tensor_entry, uFI,uFJ, i-1, tensor_dim, 0)
                F = np.zeros((len(uFI[i-1]),cores[-1].shape[-1]))
                for ir in range(len(uFI[i-1])):
                    row  = cores[0][:,uFI[i-1][ir][0],:]
                    for j in range(1,i):
                        row = row@cores[j][:,uFI[i-1][ir][j],:]
                    F[ir,:] = row[0]
            
                QF,_,_ = np.linalg.svd(F)
                qF = QF[:,:F.shape[1]]
                print(np.linalg.cond(F),np.linalg.cond(qF))
                B = np.linalg.lstsq(F,A_test,rcond = None)[0]
                cores.append(np.reshape(B,[cores[-1].shape[-1],tensor_dim[i],1]))

            else:
            
                fst_ind = np.repeat(uFI[i-1],len(uFJ[i])*tensor_dim[i],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[i]),len(uFJ[i])),len(uFI[i-1]))
                lst_ind = np.tile(uFJ[i],(len(uFI[i-1])*tensor_dim[i],1))
                ind = np.column_stack([fst_ind,mid_ind,lst_ind])
                t = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[i-1]),tensor_dim[i]*len(uFJ[i])])

                F = np.zeros((len(uFI[i-1]),cores[-1].shape[-1]))
                for ir in range(len(uFI[i-1])):
                    row  = cores[0][:,uFI[i-1][ir][0],:]
                    for j in range(1,i):
                        row = row@cores[j][:,uFI[i-1][ir][j],:]
                    F[ir,:] = row[0]
            
                QF,_,_ = np.linalg.svd(F)
                qF = QF[:,:F.shape[1]]
                print(np.linalg.cond(F),np.linalg.cond(qF))
                B = np.reshape(np.linalg.lstsq(F,t,rcond = None)[0],[cores[-1].shape[-1]*tensor_dim[i],len(uFJ[i])])

                q,r = sp.linalg.qr(B,mode = "economic")
                U,s_values,_ = np.linalg.svd(r)
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    truncation_rank[i] = cutoff
                u = q@U
                #fixed_rank = _bounded_rank(truncation_rank[i], u.shape[1])
                cores.append(np.reshape(u[:,:truncation_rank[i]],[cores[-1].shape[-1],tensor_dim[i],truncation_rank[i]]))

        if return_trunc_ranks:
            return cores, truncation_rank
        return cores

    elif orientation == "rtl":

        trunc_ranks = [len(i) for i in FIt]

        K_list = []
        L_list = []

        for i in range(len(tensor_dim)-1):
            L_list.append(tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
        for i in range(len(tensor_dim)-1):
            K_list.append(tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))

        uFJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
        uFI = [FIt[i] + K_list[i] for i in range(len(FIt))]

        core_rev_t = []
        for i in range(len(tensor_dim)-1,-1,-1):
            if i == len(tensor_dim)-1:
                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
                Q,r = sp.linalg.qr(mat,mode = "economic")
                U,s_values,_ = np.linalg.svd(r)
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    trunc_ranks[i-1] = cutoff
                q = Q@U
                core_rev_t.append(np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],1]))
            else:
                F = np.zeros((core_rev_t[0].shape[0],len(uFJ[i])))
                for j in range(len(uFJ[i])):
                    ind = list(uFJ[i][j])
                    col = core_rev_t[0][:,ind[0],:]
                    for l in range(1,len(core_rev_t)):
                        col = col@core_rev_t[l][:,ind[l],:]
                    F[:,[j]] = col

                mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
                Ft = np.ascontiguousarray(F.T)
                matt = np.ascontiguousarray(mat.T)
                y = np.linalg.lstsq(Ft,matt,rcond = None)[0].T

                if i > 0:
                    y = np.reshape(y,[len(uFI[i-1]),tensor_dim[i]*core_rev_t[0].shape[0]]).T
                    Q,r = sp.linalg.qr(y,mode = "economic")
                    U,s_values,_ = np.linalg.svd(r)
                    if adap_tol is not None:
                        cutoff = rank_cutoff(s_values, adap_tol)
                        trunc_ranks[i-1] = cutoff
                    q = Q@U
                    core_rev_t.insert(0,np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],core_rev_t[0].shape[0]]))
                else:
                    core_rev_t.insert(0,np.reshape(y,[1,tensor_dim[0],core_rev_t[0].shape[0]]))

        if return_trunc_ranks:
            return core_rev_t, trunc_ranks

        return core_rev_t

    else:
        raise ValueError("orientation must be ltr or rtl")


# MPI constructors: one active MPI rank per tensor-train core.

def _cutoff(s, tol, cap):
    if tol is None:
        return cap
    if not len(s) or np.isclose(s[0], 0):
        return 1
    return max(1, int(np.count_nonzero(s / s[0] >= tol)))


def _setup(comm, dim, prepare):
    if comm is None:
        comm = MPI.COMM_WORLD
    d = len(dim)
    if d < 2 or comm.Get_size() < d:
        raise ValueError(f"Need at least d={d} MPI processes")
    rank = comm.Get_rank()
    active = comm.Split(0 if rank < d else MPI.UNDEFINED, rank)
    if rank >= d:
        return comm, None, None
    payload = prepare() if rank == 0 else None
    return comm, active, active.bcast(payload, root=0)


def _factor(mat, tol, cap):
    q, r = sp.linalg.qr(mat, mode="economic")
    u, s, _ = sp.linalg.svd(r)
    rk = min(_cutoff(s, tol, cap), u.shape[1])
    return q @ u[:, :rk]


def _finish(comm, active, core, outgoing_rank, return_trunc_ranks):
    cores = active.gather(core, root=0)
    ranks = active.gather(outgoing_rank, root=0)
    active.Free()
    if comm.Get_rank() != 0:
        return (None, None) if return_trunc_ranks else None
    return (cores, ranks[:-1]) if return_trunc_ranks else cores


def peid_par_parallel(tensor_entry, FI, FJ, sample_row, sample_col,
                                tensor_dim, adap_tol=None,
                                return_trunc_ranks=False, comm=None):
    """MPI version of peid_par with orientation="ltr", one rank per TT core."""
    d = len(tensor_dim)
    def prepare():
        fi = [[[int(x) for x in row] for row in group] for group in FI]
        fj = [[[int(x) for x in row] for row in group] for group in FJ]
        fi, fj = sample_nested(fi, fj, sample_row,
                                                   sample_col, tensor_dim, 0)
        I, _ = set_conversion(fi, fj, tensor_dim)
        return [np.asarray(x) for x in fi], [np.asarray(x) for x in fj], I
    comm, active, data = _setup(comm, tensor_dim, prepare)
    if active is None:
        return (None, None) if return_trunc_ranks else None
    k = active.Get_rank()
    fi, fj, I = data
    if k < d-1:
        mat = unfolding_submatrix(tensor_entry, fi, fj, k, tensor_dim, 1)
        u = _factor(mat, adap_tol, len(FI[k]))
        r_out = u.shape[1]
        active.send(u, dest=k+1, tag=11)
    else:
        r_out = None
    if k == 0:
        core = u.reshape(1, tensor_dim[k], r_out)
    else:
        prev = active.recv(source=k-1, tag=11)
        r_in = prev.shape[1]
        if k == d-1:
            rhs = unfolding_submatrix(tensor_entry, fi, fj,
                                                    k-1, tensor_dim, 0)
            r_out = 1
        else:
            rhs = u.reshape(len(fi[k-1]), tensor_dim[k]*r_out)
        a = sp.linalg.lstsq(prev[I[k-1], :], rhs, cond=None)[0]
        core = a.reshape(r_in, tensor_dim[k], r_out)
    return _finish(comm, active, core, r_out, return_trunc_ranks)


def peid_par2_parallel(
        tensor_entry, FI_row, FI_col, sample_row, sample_col, tensor_dim,
        adap_tol=None, return_trunc_ranks=False, comm=None):
    """MPI version of peid_par2.

    Left and right QR/SVD factors are computed concurrently by TT mode.
    Neighbors exchange factors for the least-squares solves; the center
    core uses the two boundary factors. The serial algorithm's split is
    preserved: middle mode m=floor((d-1)/2).
    """
    d = len(tensor_dim)
    m = (d-1)//2
    def prepare():
        fl, jl = sample_nested(FI_row, FI_col, sample_row,
                                                  sample_col, tensor_dim, 0)
        fr, jr = sample_nested(FI_row, FI_col, sample_row,
                                                  sample_col, tensor_dim, 1)
        il, _ = set_conversion(fl, jl, tensor_dim)
        _, ir = set_conversion(fr, jr, tensor_dim)
        return ([np.asarray(x) for x in fl], [np.asarray(x) for x in jl],
                [np.asarray(x) for x in fr], [np.asarray(x) for x in jr], il, ir)
    comm, active, data = _setup(comm, tensor_dim, prepare)
    if active is None:
        return (None, None) if return_trunc_ranks else None
    k = active.Get_rank()
    fl, jl, fr, jr, il, ir = data
    if k < m:
        mat = unfolding_submatrix(tensor_entry, fl, jl, k, tensor_dim, 1)
        u = _factor(mat, adap_tol, len(FI_row[k]))
        r_out = u.shape[1]
        active.send(u, dest=k+1, tag=21)
        if k == 0:
            core = u.reshape(1, tensor_dim[k], r_out)
        else:
            prev = active.recv(source=k-1, tag=21)
            rhs = u.reshape(len(fl[k-1]), tensor_dim[k]*r_out)
            a = sp.linalg.lstsq(prev[il[k-1], :], rhs, lapack_driver="gelsy")[0]
            core = a.reshape(prev.shape[1], tensor_dim[k], r_out)
    elif k > m:
        mat = unfolding_submatrix(tensor_entry, fr, jr,
                                                k-1, tensor_dim, 0).T
        u = _factor(mat, adap_tol, len(FI_row[k-1]))
        r_in = u.shape[1]
        active.send(u, dest=k-1, tag=22)
        if k == d-1:
            core = u.reshape(1, tensor_dim[k], r_in).transpose(2, 1, 0)
        else:
            nxt = active.recv(source=k+1, tag=22)
            rhs = u.reshape(len(jr[k]), tensor_dim[k]*r_in)
            a = sp.linalg.lstsq(nxt[ir[k], :], rhs, lapack_driver="gelsy")[0]
            core = a.reshape(nxt.shape[1], tensor_dim[k], r_in).transpose(2, 1, 0)
        r_out = core.shape[2]
    else:
        left = active.recv(source=k-1, tag=21)
        right = active.recv(source=k+1, tag=22)
        fst = np.repeat(fl[k-1], tensor_dim[k]*len(jr[k]), axis=0)
        mid = np.tile(np.repeat(np.arange(tensor_dim[k]), len(jr[k])), len(fl[k-1]))
        lst = np.tile(jr[k], (len(fl[k-1])*tensor_dim[k], 1))
        inds = np.column_stack([fst, mid, lst])
        t = np.asarray(tensor_entry(inds, tensor_dim)).reshape(
            len(fl[k-1]), tensor_dim[k]*len(jr[k]))
        t = sp.linalg.lstsq(left[il[k-1], :], t, lapack_driver="gelsy")[0]
        t = t.reshape(left.shape[1]*tensor_dim[k], len(ir[k]))
        a = sp.linalg.lstsq(right[ir[k], :], t.T, lapack_driver="gelsy")[0].T
        core = a.reshape(left.shape[1], tensor_dim[k], right.shape[1])
        r_out = right.shape[1]
    return _finish(comm, active, core, r_out, return_trunc_ranks)
