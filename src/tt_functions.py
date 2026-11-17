import math
from xml.parsers.expat import errors
import numpy as np
import itertools
import core_construct
import random
import sys
import time
try:
    from mpi4py import MPI
except Exception:
    MPI = None


# Indexing and sampling helpers shared by TTACA and PEID.


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


def refined_fiber_search(tensor_entry,tensor_dim,min_max,magnitude = False,maxiter = 10,num_samples = 10):


    inds = [[random.randint(0,i-1) for i in tensor_dim] for _ in range(num_samples)]
    
    #inds = [[int(np.floor((tensor_dim[j]/(num_samples+1))*(i+1))) for j in range(len(tensor_dim))] for i in range(num_samples)] 

    locs = []
    for ind in inds:
        for i in range(maxiter):
            prev_ind = [i for i in ind]
            
            #Check along all fibers through dimensions
            for j in range(len(tensor_dim)):
                if j==0:
                    fst_ind = np.arange(tensor_dim[j])
                    lst_ind = np.tile(ind[1:],(tensor_dim[j],1))
                    inds = np.column_stack([fst_ind,lst_ind])
                elif j==len(tensor_dim)-1:
                    fst_ind = np.tile(ind[:-1],(tensor_dim[-1],1))
                    lst_ind = np.arange(tensor_dim[j])
                    inds = np.column_stack([fst_ind,lst_ind])
                else:
                    fst_ind = np.tile(ind[:j],(tensor_dim[j],1))
                    mid_ind = np.arange(tensor_dim[j])
                    lst_ind = np.tile(ind[j+1:],(tensor_dim[j],1))
                    inds = np.column_stack([fst_ind,mid_ind,lst_ind])
                vals = tensor_entry(inds,tensor_dim)
                if min_max == "max":
                    if magnitude:
                        ind[j] = np.argmax(np.abs(vals))
                    else:
                        ind[j] = np.argmax(vals)
                elif min_max == "min":
                    if magnitude:
                        ind[j] = np.argmin(np.abs(vals))
                    else:
                        ind[j] = np.argmin(vals)
            
            #Check along all super diagonals emerging from ind using a fun little binary ordering convention
            sgn = [[0 for _ in range(len(ind)-int(np.floor(np.log2(i))+1))]+[int(b) for b in bin(i)[2:]] for i in range(1,2**len(ind),2)]
            szs = [[j for j in ind],[tensor_dim[j] - ind[j]-1 for j in range(len(ind))]]
            inds_diag = [ind]

            for k in sgn:

                #"Left side of ind"
                for l in range(min([szs[k[j]][j] for j in range(len(ind))])):
                    inds_diag.append([ind[j]+(l+1)*(2*k[j] -1) for j in range(len(ind))])
                #"Right side of ind"
                for l in range(min([szs[k[j]-1][j] for j in range(len(ind))])):
                    inds_diag.append([ind[j]+(l+1)*(-2*k[j] +1) for j in range(len(ind))])


            vals = tensor_entry(np.array(inds_diag),tensor_dim)
            if min_max == "max":
                if magnitude:
                    ind = inds_diag[np.argmax(np.abs(vals))]
                else:
                    ind = inds_diag[np.argmax(vals)]
            elif min_max == "min":
                if magnitude:
                    ind = inds_diag[np.argmin(np.abs(vals))]
                else:
                    ind = inds_diag[np.argmin(vals)]
                
            if prev_ind == ind:
                break
        if i==maxiter-1:
            print("DIDNT BREAK")
        locs.append(ind)
    values = tensor_entry(locs,tensor_dim)
    
    if min_max == "max":
        if magnitude:
            ind = locs[np.argmax(np.abs(values))]
            p = np.max(np.abs(values))
        else:
            ind = locs[np.argmax(values)]
            p = np.max(values)
    elif min_max == "min":
        if magnitude:
            ind = locs[np.argmin(np.abs(values))]
            p = np.min(np.abs(values))
        else:
            ind = locs[np.argmin(values)]
            p = np.min(values)
    FI = [[ind[:i+1]] for i in range(len(tensor_dim)-1)]
    FJ = [[ind[i+1:]] for i in range(len(tensor_dim)-1)]
    p = tensor_entry([ind],tensor_dim)[0]
    I = [[ind[i]] for i in range(len(tensor_dim)-1)]
    J = [[ind[i+1]] for i in range(len(tensor_dim)-1)]
    
    return I,J,FI,FJ,p


def ttaca_peid(tensor_entry,tensor_dim,sample_row,sample_col,min_rank = None,max_rank = None,tol = None,sampling_type = None,tol_type = "Relative",adap_toler = None,return_trunc_ranks = False):
    
    
    if sampling_type not in (None,"Par","Seq","Par2","Seq2","R-Par","R-Seq","Par-P","Par2-P"):
        raise ValueError(f"Unknown sampling_type: {sampling_type}")

    parallel = sampling_type in ("Par-P","Par2-P")
    if parallel:
        if MPI is None:
            raise ImportError("MPI sampling types require mpi4py and a working MPI installation")
        comm = MPI.COMM_WORLD
        if len(tensor_dim)<2 or comm.Get_size()<len(tensor_dim):
            raise ValueError(f"Need at least d={len(tensor_dim)} MPI processes")

    #Fix min and max rank, and tolerance
    if not min_rank:
        min_rank = [1 for _ in range(len(tensor_dim)-1)]
    else:
        if len(min_rank)!=len(tensor_dim)-1:
            raise ValueError("Length of minimum ranks is incorrect, must be an array of length:", len(tensor_dim)-1)
        
    if not max_rank:
        max_rank = [tensor_dim[i] for i in range(len(tensor_dim)-1)]
    else:
        if len(max_rank)!=len(tensor_dim)-1:
            raise ValueError("Length of maximum ranks is incorrect, must be an array of length:", len(tensor_dim)-1)
    if not tol:
        tol = [1e-6 for _ in range(len(tensor_dim)-1)]
    else:
        
        if len(tol)!=len(tensor_dim)-1:
            raise ValueError("Length of tolerances is incorrect, must be an array of length:", len(tensor_dim)-1)
        
    if not parallel or comm.Get_rank()==0:
        #print("Kickstart started")
        I,J,FI,FJ,p = refined_fiber_search(tensor_entry,tensor_dim,"max",magnitude=True)
        #print("Kickstart ended")
    
        if tol_type == "Relative":
            tol = [np.abs(p)*i for i in tol]
    
        sweep = []
        for i in range(len(tensor_dim)-1):
            if max_rank[i]>1:
                sweep.append(i)
        count = [0 for _ in range(len(tensor_dim)-1)]
        pivots = []
        Ej = [[] for _ in range(len(tensor_dim)-1)]
        D = [[] for _ in range(len(tensor_dim)-1)]
        Ei = [[] for _ in range(len(tensor_dim)-1)]
        t_start = time.time()
        for i in sweep:
        
            count[i]+=1
        
            """
            if i==1:
            
                lp = LineProfiler()
                lp.add_function(greedy_search)
                i_star,j_star,i_star_super,j_star_super,p = lp.runcall(greedy_search,tensor_entry,I,J,FI,FJ,tensor_dim,i)
                lp.print_stats()
            else:
            """
        
            i_star,j_star,i_star_super,j_star_super,p,Ej[i],D[i],Ei[i] = greedy_search(tensor_entry,I,J,FI,FJ,Ej[i],D[i],Ei[i],tensor_dim,i)
            #i_star,j_star,i_star_super,j_star_super,p = greedy_search(tensor_entry,I,J,FI,FJ,tensor_dim,i)
        
            I[i].append(i_star_super)
            J[i].append(j_star_super)
            FI[i].append(i_star)
            FJ[i].append(j_star)
        
        
            if count[i]<max_rank[i]-1 and np.abs(p)>tol[i]:
            
                sweep.append(i)


        FI = [[[int(i) for i in FI[l][j]] for j in range(len(FI[l]))] for l in range(len(tensor_dim)-1)]
        FJ = [[[int(i) for i in FJ[l][j]] for j in range(len(FJ[l]))] for l in range(len(tensor_dim)-1)]
        trunc_ranks = [len(i) for i in FI]
    if parallel:
        payload = (FI,FJ,trunc_ranks) if comm.Get_rank()==0 else None
        FI,FJ,trunc_ranks = comm.bcast(payload,root = 0)

    #print("Time to find pivots",time.time() - t_start,flush = True)
    if sampling_type is None:
        cores_2 = core_construct.ttaca_cores(tensor_entry, FI, FJ, trunc_ranks, tensor_dim, {})
    elif sampling_type == "Par":
        cores_2 = core_construct.peid_par(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
    elif sampling_type == "Seq":
        cores_2 = core_construct.peid_seq(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
    elif sampling_type == "Par2":
        cores_2 = core_construct.peid_par2(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
    elif sampling_type == "Seq2":
        cores_2 = core_construct.peid_seq2(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
    elif sampling_type == "R-Par":
        cores_2 = core_construct.peid_r(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,base = "par")
    elif sampling_type == "R-Seq":
        cores_2 = core_construct.peid_r(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,base = "seq")
    elif sampling_type == "Par-P":
        cores_2 = core_construct.peid_par_parallel(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler,comm = comm)
    elif sampling_type == "Par2-P":
        cores_2 = core_construct.peid_par2_parallel(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler,comm = comm)
    if return_trunc_ranks:
        return cores_2,FI,FJ,trunc_ranks
    return cores_2,FI,FJ


def greedy_search(tensor_entry,I,J,FI,FJ,Ej,D,Ei,tensor_dim,k,tol=0,sample_size = 100,maxiter = 40):
    flag = False
    
    if k==0:
        if len(FI[k])==tensor_dim[k] or len(FJ[k])==len(FJ[k+1])*tensor_dim[k+1]:
            flag = True
    elif k==len(tensor_dim)-2:
        if len(FI[k])==len(FI[k-1])*tensor_dim[k] or len(FJ[k])==tensor_dim[k+1]:
            flag = True
    else:
        if len(FI[k])==len(FI[k-1])*tensor_dim[k] or len(FJ[k])==len(FJ[k+1])*tensor_dim[k+1]:
            flag = True

    if flag:
        return FI,FJ,0
    
    else:
        
        #Initial sample
        unavail_row = list(I[k])
        unavail_col = list(J[k])
        
        if k==0:
            s_row = cheap_sample(0, tensor_dim[k], unavail_row, sample_size)
            s_col = cheap_sample(0, tensor_dim[k + 1] * len(J[k + 1]), unavail_col, sample_size)
            t_row = [i for i in s_row]
            t_col = [tuple(reversed(large_unravel(i, [len(FJ[k + 1]), tensor_dim[k + 1]]))) for i in s_col]
            samples = [[t_row[i]]+[t_col[i][0]] + FJ[k+1][t_col[i][1]] for i in range(min(len(t_row),len(t_col)))]
        elif k==len(tensor_dim)-2:
            s_row = cheap_sample(0, len(I[k - 1]) * tensor_dim[k], unavail_row, sample_size)
            s_col = cheap_sample(0, tensor_dim[k + 1], unavail_col, sample_size)
            t_row = [large_unravel(i, [len(FI[k - 1]), tensor_dim[k]]) for i in s_row]
            t_col = [i for i in s_col]
            
            samples = [FI[k-1][t_row[i][0]]+[t_row[i][1]]+[t_col[i]] for i in range(min(len(t_row),len(t_col)))]
        else:
            s_row = cheap_sample(0, len(I[k - 1]) * tensor_dim[k], unavail_row, sample_size)
            s_col = cheap_sample(0, tensor_dim[k + 1] * len(J[k + 1]), unavail_col, sample_size)
            t_row = [large_unravel(i, [len(FI[k - 1]), tensor_dim[k]]) for i in s_row]
            t_col = [tuple(reversed(large_unravel(i, [len(FJ[k + 1]), tensor_dim[k + 1]]))) for i in s_col]
            
            samples = [FI[k-1][t_row[i][0]]+[t_row[i][1]]+[t_col[i][0]] + FJ[k+1][t_col[i][1]] for i in range(min(len(t_row),len(t_col)))]
        

        vals = tensor_entry(samples,tensor_dim)

        #Selection
        index = np.argmax(np.abs(vals))
        i_star = np.array(samples[index][:k+1])
        j_star = np.array(samples[index][k+1:])
        p = vals[index]

        comp_start = time.time()
        #loop
        if len(Ej) ==0:
            C = unfolding_submatrix(tensor_entry,FI,FJ,k,tensor_dim,1)
            R = unfolding_submatrix(tensor_entry,FI,FJ,k,tensor_dim,0)

            Ej = np.zeros_like(C)
            Ei = np.zeros_like(R)
            D = np.zeros((len(I[k])))
            Ej[:,0] = C[:,0]
            Ei[0,:] = R[0,:]
            D[0] = 1/C[I[k][0],0]

            for i in range(1,len(I[k])):
                ej = C[:,i] - Ej@np.diag(D)@Ei[:,J[k][i]]
                ei = R[i,:] - Ej[I[k][i],:]@np.diag(D)@Ei
                invdelt = ej[I[k][i]]

                if invdelt==0:   
                    break
                else:
                    Ej[:,i] = ej
                    Ei[i,:] = ei
                    D[i] = 1/invdelt
        else:
            #A flag is need to account for when neighboring dimensions are not added to on previous pass.
            FJ_end = [[FJ[k][-1]] for k in range(len(FJ))]
            FI_end = [[FI[k][-1]] for k in range(len(FI))]
            c_t = unfolding_submatrix(tensor_entry,FI,FJ_end,k,tensor_dim,1)
            r_t = unfolding_submatrix(tensor_entry,FI_end,FJ,k,tensor_dim,0)
            if k==0:
                R_t = unfolding_submatrix(tensor_entry,FI,FJ,k,tensor_dim,0,full = False)
                

                if r_t.shape[1]!=Ei.shape[1]:
                    Ei_partial = np.zeros((R_t.shape[0]-1,R_t.shape[1]))
                
                    Ei_partial[0,:] = R_t[0,:]
                    for i in range(1,len(I[k])-1):
                        
                        Ei_partial[i,:] = R_t[i,:] - Ej[I[k][i],:]@np.diag(D)@Ei_partial
                    
                    Ei = np.hstack((Ei,Ei_partial))
                
            elif k==len(tensor_dim)-2:
                C_t = unfolding_submatrix(tensor_entry,FI,FJ,k,tensor_dim,1,full = False)


                if c_t.shape[0]!=Ej.shape[0]:
                    Ej_partial = np.zeros((C_t.shape[0],C_t.shape[1]-1))
                    Ej_partial[:,0] = C_t[:,0]

                    for i in range(1,len(I[k])-1):
                        Ej_partial[:,i] = C_t[:,i] - Ej_partial@np.diag(D)@Ei[:,J[k][i]]
                    Ej = np.vstack((Ej,Ej_partial))


            else:

                C_t = unfolding_submatrix(tensor_entry,FI,FJ,k,tensor_dim,1,full = False)
                R_t = unfolding_submatrix(tensor_entry,FI,FJ,k,tensor_dim,0,full = False)


                if c_t.shape[0]!=Ej.shape[0]:
                    Ej_partial = np.zeros((C_t.shape[0],C_t.shape[1]-1))
                    Ej_partial[:,0] = C_t[:,0]

                    for i in range(1,len(I[k])-1):
                        Ej_partial[:,i] = C_t[:,i] - Ej_partial@np.diag(D)@Ei[:,J[k][i]]
                    Ej = np.vstack((Ej,Ej_partial))
                    
                if r_t.shape[1]!=Ei.shape[1]:
                    Ei_partial = np.zeros((R_t.shape[0]-1,R_t.shape[1]))
                
                    Ei_partial[0,:] = R_t[0,:]
                    for i in range(1,len(I[k])-1):
                        
                        Ei_partial[i,:] = R_t[i,:] - Ej[I[k][i],:]@np.diag(D)@Ei_partial
                    
                    Ei = np.hstack((Ei,Ei_partial))

            ej = c_t - Ej@np.diag(D)@Ei[:,[J[k][-1]]]
            ei = r_t - Ej[I[k][-1],:]@np.diag(D)@Ei
            invdelt = ej[I[k][-1],0]
            if invdelt!=0:
                Ej = np.hstack((Ej,ej))
                Ei = np.vstack((Ei,ei))
                D = np.append(D,1/invdelt)


        for l in range(maxiter):

            #Compute row values
            if k==0:

                fst_indt = np.tile(i_star,(tensor_dim[k+1]*len(FJ[k+1]),1))
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(FJ[k+1]))
                lst_indt = np.repeat(FJ[k+1],tensor_dim[k+1],axis = 0)
                lists = [fst_indt,mid_indt,lst_indt]
                ind = np.column_stack(lists)
            elif k==len(tensor_dim)-2:
                fst_indt = np.tile(i_star,(tensor_dim[k+1],1))
                lst_indt = np.arange(tensor_dim[k+1])
                lists = [fst_indt,lst_indt]
                ind = np.column_stack(lists)
            else:
                fst_indt = np.tile(i_star,(tensor_dim[k+1]*len(FJ[k+1]),1))
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(FJ[k+1]))
                lst_indt = np.repeat(FJ[k+1],tensor_dim[k+1],axis = 0)
                lists = [fst_indt,mid_indt,lst_indt]

                ind = np.column_stack(lists)

            #Translate i_star to superblock location
            count = 0
            if k==0:
                i_star_super = i_star[0]
            else:
                for i in range(len(FI[k-1])):
                    
                    if list(i_star[:-1])==FI[k-1][i]:
                        i_star_super = large_ravel([i, i_star[-1]], [len(I[k - 1]), tensor_dim[k]])
                        break

            row_val = tensor_entry(ind,tensor_dim) - Ej[i_star_super,:]@np.diag(D)@Ei

            #sorted_inds = np.flip(inds[np.argsort(row_val[inds])])
            sorted_inds = np.flip(np.argsort(np.abs(row_val)))
            for j in sorted_inds:
                if j not in unavail_col:
                    p = row_val[j]
                    break

            j_star = list(ind[j][k+1:])
            

            #Compute column values
            if k==0:
                fst_ind =np.arange(tensor_dim[k])
                lst_ind = np.tile(j_star,(tensor_dim[k],1))
                lists = [fst_ind,lst_ind]
                ind = np.column_stack(lists)

            elif k == len(tensor_dim)-2:
                fst_ind = np.repeat(FI[k-1],tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.arange(tensor_dim[k]),len(FI[k-1]))
                lst_ind = np.tile(j_star,(len(FI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

            else:

                fst_ind = np.repeat(FI[k-1],tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.arange(tensor_dim[k]),len(FI[k-1]))
                lst_ind = np.tile(j_star,(len(FI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)


            if k==len(tensor_dim)-2:
                j_star_super = j_star[0]
            else:
                for i in range(len(FJ[k+1])):
                    
                    if list(j_star[1:])==FJ[k+1][i]:
                        j_star_super = large_ravel([i, j_star[0]], [len(J[k + 1]), tensor_dim[k + 1]])
                        break

            col_val = tensor_entry(ind,tensor_dim) - Ej@np.diag(D)@Ei[:,j_star_super]

 
            sorted_inds = np.flip(np.argsort(np.abs(col_val)))

            for j in sorted_inds:
                if j not in unavail_row:
                    p = col_val[j]
                    break
            i_star_temp = ind[j][:k+1]

            if all(i_star == i_star_temp):
                break
            else:
                i_star = list(i_star_temp)

        #One last conversion to assure they are accurate
        if k==0:
            i_star_super = i_star[0]
        else:
            for i in range(len(FI[k-1])):
                if list(i_star[:-1])==FI[k-1][i]:
                    i_star_super = large_ravel([i, i_star[-1]], [len(I[k - 1]), tensor_dim[k]])

        if k==len(tensor_dim)-2:
            j_star_super = j_star[0]
        else:
            for i in range(len(FJ[k+1])):
                if list(j_star[1:])==FJ[k+1][i]:
                    j_star_super = large_ravel([i, j_star[0]], [len(J[k + 1]), tensor_dim[k + 1]])

        return list(i_star),list(j_star),i_star_super,j_star_super,p,Ej,D,Ei
