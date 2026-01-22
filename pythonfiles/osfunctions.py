import numpy as np
import scipy as sp
import math
import samplingfunctions
import tt_functions
import indexingfunctions
try:
    from mpi4py import MPI
except Exception:
    MPI = None






def two_sided_nested_constructionv2(tensor_entry,FI_row,FI_col,sample_row,sample_col,trunc_ranks,tensor_dim):
   
    cores_left = []
    uj_left = []
   
    cores_right_rev = []
    uj_right = []
   
    d = len(tensor_dim)
    n = len(trunc_ranks)
    mid_l = math.floor((d-1)/2)
    if not (d-1)%2:
        mid_r = math.floor((d-1)/2)
        flag = True
    else:
        mid_r = math.floor((d-1)/2)
        flag = False

    FI_left,FJ_left = samplingfunctions.sample_nested(FI_row, FI_col, sample_row, sample_col, tensor_dim,0)
    I_left,J_left = indexingfunctions.set_conversion(FI_left, FJ_left, tensor_dim)
    
    FI_right,FJ_right = samplingfunctions.sample_nested(FI_row, FI_col, sample_row, sample_col, tensor_dim,1)
    I_right,J_right = indexingfunctions.set_conversion(FI_right, FJ_right, tensor_dim)




    cores_right_rev = [np.zeros((1,tensor_dim[i],trunc_ranks[i-1])) if i==len(trunc_ranks) else np.zeros((trunc_ranks[i],tensor_dim[i],trunc_ranks[i-1])) for i in range(len(trunc_ranks),mid_r,-1)]

    cores_left = [np.zeros((1,tensor_dim[i],trunc_ranks[i])) if i==0 else np.zeros((trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i])) for i in range(mid_l)]
    
    for i in range(n,mid_r,-1):
        if i == n:            
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_right, FJ_right, i-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            #qq,rr = np.linalg.qr(mat)
            #q_r,r_r,p = np.linalg.svd(rr)
            #u = qq@q_r
            
            uj_right.append(u[:,:trunc_ranks[i-1]])
            uj_tilde = np.reshape(uj_right[-1],[trunc_ranks[i-1],tensor_dim[i]])
            cores_right_rev[n-i] = np.reshape(uj_right,[1,tensor_dim[i],trunc_ranks[i-1]])
        elif i==0:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_right, FJ_right, 0, tensor_dim, 1).T
            A = sp.linalg.lstsq(uj_right[-1][J_right[0],:],mat,lapack_driver="gelsy")[0]
            cores_right_rev.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],1]))
        else:
            
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_right, FJ_right, i-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj_right.append(u[:,:trunc_ranks[i-1]])
            uj_tilde = np.reshape(uj_right[-1],[len(FJ_right[i]),tensor_dim[i]*trunc_ranks[i-1]])  

            A = sp.linalg.lstsq(uj_right[-2][J_right[i],:],uj_tilde,lapack_driver="gelsy")[0]
            cores_right_rev[n-i] = np.reshape(A,[trunc_ranks[i],tensor_dim[i],trunc_ranks[i-1]])
      
    cores_right = [np.transpose(cores_right_rev[i],[2,1,0]) for i in range(len(cores_right_rev)-1,-1,-1)]
   
    for i in range(mid_l):
        if i == n:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_left, FJ_left, i-1, tensor_dim, 0)
            A = sp.linalg.lstsq(uj_left[-1][I_left[i-1],:],mat,lapack_driver="gelsy")[0]
            cores_left[i] = np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],1])
        elif i==0:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_left, FJ_left, i, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj_left.append(u[:,:trunc_ranks[i]])
            cores_left[i] = np.reshape(uj_left[-1],[1,tensor_dim[i],trunc_ranks[i]])
        else:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_left, FJ_left, i, tensor_dim, 1)
            
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj_left.append(u[:,:trunc_ranks[i]])
            uj_tilde = np.reshape(uj_left[-1],[len(FI_left[i-1]),tensor_dim[i]*trunc_ranks[i]])
            A = sp.linalg.lstsq(uj_left[-2][I_left[i-1],:],uj_tilde,lapack_driver="gelsy")[0]
            cores_left[i] = np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]])
    
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
   
    return cores_all

def two_sided_nested_construction_parallel(tensor_entry,FI_row,FI_col,sample_row,sample_col,trunc_ranks,tensor_dim):
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank < len(tensor_dim):
        color = 1
        key = rank
    else:
        color = 2
        key = rank
    subcomm = comm.Split(color,key)
    subrank = subcomm.Get_rank()
    subsize = subcomm.Get_size()


    if color == 1:
        cores_left = []
        uj_left = []
    
        cores_right_rev = []
        uj_right = []
    
        d = len(tensor_dim)
        n = len(trunc_ranks)
        mid_l = math.floor((d-1)/2)
        if not (d-1)%2:
            mid_r = math.floor((d-1)/2)
            flag = True
        else:
            mid_r = math.floor((d-1)/2)
            flag = False

        if subrank == 0:
            FI_left,FJ_left = samplingfunctions.sample_nested(FI_row, FI_col, sample_row, sample_col, tensor_dim,0)
            I_left,J_left = indexingfunctions.set_conversion(FI_left, FJ_left, tensor_dim)
            
            FI_right,FJ_right = samplingfunctions.sample_nested(FI_row, FI_col, sample_row, sample_col, tensor_dim,1)
            I_right,J_right = indexingfunctions.set_conversion(FI_right, FJ_right, tensor_dim)
        else:
            FI_left = None
            FJ_left = None
            I_left = None
            J_left = None
            FI_right = None
            FJ_right = None
            I_right = None
            J_right = None
        FI_left = subcomm.bcast(FI_left,root = 0)
        FJ_left = subcomm.bcast(FJ_left,root = 0)
        I_left = subcomm.bcast(I_left,root = 0)
        J_left = subcomm.bcast(J_left,root = 0)
        FI_right = subcomm.bcast(FI_right,root = 0)
        FJ_right = subcomm.bcast(FJ_right,root = 0)
        I_right = subcomm.bcast(I_right,root = 0)
        J_right = subcomm.bcast(J_right,root = 0)

        subcomm.barrier()
        
        #for i in range(n,mid_r,-1):
        if subrank == n:         
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_right, FJ_right, subrank-1, tensor_dim, 0).T

            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj_right = u[:,:trunc_ranks[subrank-1]]

            subcomm.send(uj_right,dest = subrank-1,tag = 11)
            uj_tilde = np.reshape(uj_right,[trunc_ranks[subrank-1],tensor_dim[subrank]])
            cores_right_rev = np.reshape(uj_right,[1,tensor_dim[subrank],trunc_ranks[subrank-1]])
            
        elif subrank > mid_r and subrank < n:
            
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_right, FJ_right, subrank-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj_right = u[:,:trunc_ranks[subrank-1]]

            if subrank >mid_r + 1:
                subcomm.send(uj_right,dest = subrank-1,tag = 11)
            
            uj_right_prev = subcomm.recv(source = subrank+1,tag = 11)
            uj_tilde = np.reshape(uj_right,[len(FJ_right[subrank]),tensor_dim[subrank]*trunc_ranks[subrank-1]])  
            A = np.linalg.lstsq(uj_right_prev[J_right[subrank],:],uj_tilde,rcond = None)[0]
            cores_right_rev = np.reshape(A,[trunc_ranks[subrank],tensor_dim[subrank],trunc_ranks[subrank-1]])


        cores_right_list = subcomm.gather(cores_right_rev,root = 0) 
        if subrank == 0:    
            cores_right = [np.transpose(cores_right_list[i],[2,1,0]) for i in range(mid_r+1,n+1)]
    

        if subrank==0:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_left, FJ_left, subrank, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj_left = u[:,:trunc_ranks[subrank]]
            if mid_l>1:
                subcomm.send(uj_left,dest = subrank+1,tag = 77)
            cores_left = np.reshape(uj_left,[1,tensor_dim[subrank],trunc_ranks[subrank]])
        elif subrank >0 and subrank < mid_l:
             
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI_left, FJ_left, subrank, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj_left = u[:,:trunc_ranks[subrank]]

            if subrank < mid_l -1:
                subcomm.send(uj_left,dest = subrank+1,tag = 77)
                
            uj_left_prev = subcomm.recv(source = subrank-1,tag = 77)
            uj_tilde = np.reshape(uj_left,[len(FI_left[subrank-1]),tensor_dim[subrank]*trunc_ranks[subrank]])
            A = np.linalg.lstsq(uj_left_prev[I_left[subrank-1],:],uj_tilde,rcond = None)[0]
            cores_left = np.reshape(A,[trunc_ranks[subrank-1],tensor_dim[subrank],trunc_ranks[subrank]])
        
        subcomm.barrier()   
        cores_left_list = subcomm.gather(cores_left,root = 0)

        if mid_l -1 >0:
            if subrank == mid_l-1:
                subcomm.send(uj_left,dest = 0,tag = 70)
        if subrank == mid_r + 1:
            subcomm.send(uj_right,dest = 0,tag = 120)

        
        if subrank ==0:
            if mid_l -1 >0:
                uj_left_prev = subcomm.recv(source = mid_l-1,tag = 70)
            else:
                uj_left_prev = uj_left
            uj_right_prev = subcomm.recv(source = mid_r + 1,tag = 120)
            #inds = [i+math.prod(tensor_dim[mid_l+1:])*(j) for j in range(tensor_dim[mid_l]) for i in FJ_right[mid_l]]
            #full_inds = [tuple(indexingfunctions.Large_unravel(i,tensor_dim[:mid_l])) + tuple(indexingfunctions.Large_unravel(j,tensor_dim[mid_l:])) for i in FI_left[mid_l-1] for j in inds]
            #t = np.reshape(tensor_entry(full_inds,tensor_dim),[len(FI_left[mid_l-1]),tensor_dim[mid_l]*len(FJ_right[mid_l])])
            fst_ind = np.repeat(FI_left[mid_l-1],tensor_dim[mid_l]*len(FJ_right[mid_l]),axis = 0)
            mid_ind = np.tile(np.repeat(np.arange(tensor_dim[mid_l]),len(FJ_right[mid_l])),len(FI_left[mid_l-1]))
            lst_ind = np.tile(FJ_right[mid_l],(len(FI_left[mid_l-1])*tensor_dim[mid_l],1))
            ind_t = np.column_stack([fst_ind,mid_ind,lst_ind])
            t = np.reshape(tensor_entry(ind_t,tensor_dim),[len(FI_left[mid_l-1]),tensor_dim[mid_l]*len(FJ_right[mid_l])])
            t = np.linalg.lstsq(uj_left_prev[I_left[mid_l-1],:],t,rcond = None)[0]
            t = np.reshape(t,[trunc_ranks[mid_l-1]*tensor_dim[mid_l],len(J_right[mid_l])])

            r = np.linalg.lstsq(uj_right_prev[J_right[mid_l],:],t.T,rcond = None)[0].T
            t1 = np.reshape(r,[trunc_ranks[mid_l-1],tensor_dim[mid_l],trunc_ranks[mid_l]])
    
            cores_all = cores_left_list[:mid_l]
            cores_all.append(t1)
            for i in cores_right:
                cores_all.append(i)
    
            return cores_all


def two_sided_nonnested_constructionv2(tensor_entry,FIt,FJt,sample_row,sample_col,trunc_ranks,tensor_dim):
   
    cores = []
    cores_left = []
    uj_left = []
   
    cores_right = []
    uj_right = []
   
    d = len(tensor_dim)
    n = len(trunc_ranks)
    mid_l = math.floor((d-1)/2)
    if not (d-1)%2:
        mid_r = math.floor((d-1)/2)
        flag = True
    else:
        mid_r = math.floor((d-1)/2)
        flag = False
        
    K_list = []
    L_list = []

    for i in range(len(tensor_dim)-1):
        L_list.append(samplingfunctions.tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
    for i in range(len(tensor_dim)-1):
        K_list.append(samplingfunctions.tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))
 
    FJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
    FI = [FIt[i] + K_list[i] for i in range(len(FIt))]
    
        
    for i in range(n,mid_r,-1):
        #print(i,mid_r,mid_l)
        if i == len(tensor_dim)-1:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI,FJ, i-1, tensor_dim, 0).T
            Q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            q = Q@U
            
            
            cores_right.append(np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],1]))
        else:
            G = np.zeros((trunc_ranks[i],len(FJ[i])))
            for j in range(len(FJ[i])):
                ind = list(FJ[i][j])#indexingfunctions.Large_unravel(FJ[i][j],tensor_dim[i+1:])
                col = cores_right[0][:,ind[0],:]
                for l in range(1,len(cores_right)):
                    col = col@cores_right[l][:,ind[l],:]
                G[:,[j]] = col
            
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI,FJ, i, tensor_dim, 1)
            y = sp.linalg.lstsq(G.T,mat.T,lapack_driver="gelsy")[0].T
            
            
            if i>0:
                y= np.reshape(y,[len(FI[i-1]),tensor_dim[i]*trunc_ranks[i]]).T
                Q,r = sp.linalg.qr(y,mode = "economic")
                U,_,_ = np.linalg.svd(r)
                q = Q@U
                
                cores_right.insert(0,np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
            else:
                cores_right.insert(0,np.reshape(y,[1,tensor_dim[0],trunc_ranks[0]]))
    
    for i in range(mid_l):
        if i==0:
            B_test = tt_functions.unfolding_submatrix(tensor_entry, FI,FJ, i, tensor_dim, 1)
            Q,r = sp.linalg.qr(B_test,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            q = Q@U
            
            cores_left.append(np.reshape(q[:,:trunc_ranks[i]],[1,tensor_dim[i],trunc_ranks[i]]))
        elif i==len(tensor_dim)-1:
            K = K_list[-1]
            ind_row = [indexingfunctions.Large_unravel(ir,tensor_dim[:i]) for ir in list(tuple(FI[i-1]))]

            A_test = tt_functions.unfolding_submatrix(tensor_entry, FI,FJ, i-1, tensor_dim, 0)
            F = []
            for ir in ind_row:
                row  = cores_left[0][:,ir[0],:]
                for j in range(1,i):
                    row = row@cores_left[j][:,ir[j],:]
                F.append(row[0])
            F = np.array(F)
            B = sp.linalg.lstsq(F,A_test,lapack_driver="gelsy")[0]
            cores_left.append(np.reshape(B,[trunc_ranks[i-1],tensor_dim[i],1]))
        else:
            K = K_list[i-1]
            L = L_list[i]
            ind_row = list(FI[i-1])#[indexingfunctions.Large_unravel(ir,tensor_dim[:i]) for ir in list(tuple(FI[i-1]))]
            A_test = tt_functions.unfolding_submatrix(tensor_entry, FI,FJ, i-1, tensor_dim, 0)
           
            #inds = [l+math.prod(tensor_dim[i+1:])*(j) for j in range(tensor_dim[i]) for l in FJ[i]]
            #full_inds = [tuple(indexingfunctions.Large_unravel(l,tensor_dim[:i])) + tuple(indexingfunctions.Large_unravel(j,tensor_dim[i:])) for l in FI[i-1] for j in inds]
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
            U,_,_ = np.linalg.svd(r)
            q = Q@U
            cores_left.append(np.reshape(q[:,:trunc_ranks[i]],[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
   
    i=mid_l
    ind_row = list(FI[i-1])#[indexingfunctions.Large_unravel(ir,tensor_dim[:i]) for ir in list(tuple(FI[i-1]))]

    F = []
    for ir in ind_row:
        row  = cores_left[0][:,ir[0],:]
        for j in range(1,i):
            row = row@cores_left[j][:,ir[j],:]
        F.append(row[0])
    F = np.array(F)
    
    #print(len(cores_right),i)
    G = np.zeros((trunc_ranks[i],len(FJ[i])))
    for j in range(len(FJ[i])):
        ind = list(FJ[i][j])#indexingfunctions.Large_unravel(FJ[i][j],tensor_dim[i+1:])
        col = cores_right[0][:,ind[0],:]
        for l in range(1,len(cores_right)):
            col = col@cores_right[l][:,ind[l],:]
        G[:,[j]] = col
    
    #inds = [l+math.prod(tensor_dim[i+1:])*(j) for j in range(tensor_dim[i]) for l in FJ[i]]
    #full_inds = [tuple(indexingfunctions.Large_unravel(l,tensor_dim[:i])) + tuple(indexingfunctions.Large_unravel(j,tensor_dim[i:])) for l in FI[i-1] for j in inds]
    
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
    return cores

def rtl_nested_construction(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim):
   
    cores = []
    uj = []
    uj_tilde = []
    n = len(trunc_ranks)
    superblock_time = 0

    uFI,uFJ = samplingfunctions.sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,1)
    uK,uL = indexingfunctions.set_conversion(uFI,uFJ, tensor_dim)
    I = uK
    J = uL

    uFI = [np.array(i) for i in uFI]
    uFJ = [np.array(i) for i in uFJ]

    

    
   
    for i in range(n,-1,-1):
        if i == n:  
            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            
            uj.append(u[:,:trunc_ranks[i-1]])
            uj_tilde = np.reshape(uj[-1],[trunc_ranks[i-1],tensor_dim[i]])
            cores.append(np.reshape(uj,[1,tensor_dim[i],trunc_ranks[i-1]]))
        elif i==0:

            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, 0, tensor_dim, 1).T
            A = sp.linalg.lstsq(uj[-1][J[0],:],mat,lapack_driver="gelsy")[0]
            cores.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],1]))
        else:
            
            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            
            uj.append(u[:,:trunc_ranks[i-1]])
            uj_tilde = np.reshape(uj[-1],[len(uFJ[i]),tensor_dim[i]*trunc_ranks[i-1]])  
            A = sp.linalg.lstsq(uj[-2][J[i],:],uj_tilde,lapack_driver="gelsy")[0]
            cores.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],trunc_ranks[i-1]]))            

    core_list = [np.transpose(cores[i],[2,1,0]) for i in range(len(tensor_dim)-1,-1,-1)]
    return core_list


def rtl_nested_construction_parallel(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim):
   

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank < len(tensor_dim):
        color = 1
        key = rank
    else:
        color = 2
        key = rank
    subcomm = comm.Split(color,key)
    subrank = subcomm.Get_rank()
    subsize = subcomm.Get_size()

    if color ==1:

        cores = []
        
        uj = []
        uj_tilde = []
        n = len(trunc_ranks)
        superblock_time = 0

        if subrank == 0:
            FI,FJ = samplingfunctions.sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,1)
            uK,uL = indexingfunctions.set_conversion(FI,FJ, tensor_dim)
            I = uK
            J = uL

            FI = [np.array(i) for i in FI]
            FJ = [np.array(i) for i in FJ]
        else:
            FI = None
            FJ = None
            I = None
            J = None
        FI = subcomm.bcast(FI,root=0)
        FJ = subcomm.bcast(FJ,root=0)
        I = subcomm.bcast(I,root=0)
        J = subcomm.bcast(J,root=0)
    
        
        if subrank == n:  
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI, FJ, subrank-1, tensor_dim, 0).T
            Q,r = sp.linalg.qr(mat,mode = "economic")
            u,_,_ = np.linalg.svd(r)
            q = Q@u
            uj = q[:,:trunc_ranks[subrank - 1]]
            #if trunc_ranks[subrank-1] >= min(mat.shape):
            #    uj,_ = np.linalg.qr(mat)
            #else:
            #    uj, _, _ = sp.sparse.linalg.svds(mat, k=trunc_ranks[subrank-1])
            subcomm.send(uj,dest = subrank-1,tag = 77)
            uj_tilde = np.reshape(uj,[trunc_ranks[subrank-1],tensor_dim[subrank]])
            cores=np.reshape(uj,[1,tensor_dim[subrank],trunc_ranks[subrank-1]])
        elif subrank==0:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI, FJ, 0, tensor_dim, 1).T
            uj_prev = subcomm.recv(source = subrank+1,tag = 77)
            A = np.linalg.lstsq(uj_prev[J[0],:],mat,rcond = None)[0]
            cores=np.reshape(A,[trunc_ranks[subrank],tensor_dim[subrank],1])
        else:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI, FJ, subrank-1, tensor_dim, 0).T
            Q,r = sp.linalg.qr(mat,mode = "economic")
            u,_,_ = np.linalg.svd(r)
            q = Q@u
            #if trunc_ranks[subrank-1] >= min(mat.shape):
            #    uj,_ = np.linalg.qr(mat)
            #else:
            #    uj, _, _ = sp.sparse.linalg.svds(mat, k=trunc_ranks[subrank-1])
            uj = q[:,:trunc_ranks[subrank-1]]
            subcomm.send(uj,dest = subrank-1,tag = 77)
            uj_tilde = np.reshape(uj,[len(FJ[subrank]),tensor_dim[subrank]*trunc_ranks[subrank-1]]) 
            uj_prev = subcomm.recv(source = subrank+1,tag = 77) 
            A = np.linalg.lstsq(uj_prev[J[subrank],:],uj_tilde,rcond = None)[0]
            cores=np.reshape(A,[trunc_ranks[subrank],tensor_dim[subrank],trunc_ranks[subrank-1]])           
        core_list = subcomm.gather(cores,root=0)
        if subrank ==0:
            cores_all = [np.transpose(i,[2,1,0]) for i in core_list]
            return cores_all

def ltr_nested_construction(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim):

    cores = []
    uj = []
    uj_tilde = []
    n = len(trunc_ranks)

    FI = [[[int(i) for i in FI[j][k]] for k in range(len(FI[j]))] for j in range(len(FI))]
    FJ = [[[int(i) for i in FJ[j][k]] for k in range(len(FJ[j]))] for j in range(len(FJ))]

    uFI,uFJ = samplingfunctions.sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,0)
    uK,uL = indexingfunctions.set_conversion(uFI, uFJ, tensor_dim)


    I = uK
    J = uL

    uFI = [np.array(i) for i in uFI]
    uFJ = [np.array(i) for i in uFJ]



    for i in range(n+1):
        if i == n:
            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0)            
            
            A = np.linalg.lstsq(uj[-1][I[i-1],:],mat,rcond = None)[0]
            cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],1]))
        elif i==0:
            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj.append(u[:,:trunc_ranks[i]])
            cores.append(np.reshape(uj[0],[1,tensor_dim[i],trunc_ranks[i]]))
        else:
           
            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj.append(u[:,:trunc_ranks[i]])

            
            uj_tilde = np.reshape(uj[i],[len(uFI[i-1]),tensor_dim[i]*trunc_ranks[i]])

            
            A = np.linalg.lstsq(uj[i-1][I[i-1],:],uj_tilde,rcond = None)[0]

            cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
    
    return cores

def ltr_nested_construction_parallel(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim):

    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank < len(tensor_dim):
        color = 1
        key = rank
    else:
        color = 2
        key = rank
    subcomm = comm.Split(color,key)
    subrank = subcomm.Get_rank()
    subsize = subcomm.Get_size()

    if color ==1:
        cores_all = []
        uj = []
        uj_tilde = []
        n = len(trunc_ranks)
        
        if subrank==0:

            FI,FJ = samplingfunctions.sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,0)
            uK,uL = indexingfunctions.set_conversion(FI, FJ, tensor_dim)

            I = uK
            J = uL


        else:
            FI = None
            FJ = None
            I = None
            J = None
        FI = subcomm.bcast(FI, root=0)
        FJ = subcomm.bcast(FJ, root=0)
        I = subcomm.bcast(I, root=0)
        J = subcomm.bcast(J, root=0)


        if subrank==0:
            
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI, FJ, subrank, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj = u[:,:trunc_ranks[subrank]]
            #if trunc_ranks[subrank] >= min(mat.shape):
            #    uj,_ = np.linalg.qr(mat)
            #else:
            #    uj, _, _ = sp.sparse.linalg.svds(mat, k=trunc_ranks[subrank])
            
            subcomm.send(uj,dest=subrank+1,tag=11)
            cores = np.reshape(uj,[1,tensor_dim[subrank],trunc_ranks[subrank]])
            

        elif subrank==n:
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI, FJ, subrank-1, tensor_dim, 0)
            uj_prev = subcomm.recv(source=subrank-1,tag=11)
            A = np.linalg.lstsq(uj_prev[I[subrank-1],:],mat,rcond = None)[0]
            cores = np.reshape(A,[trunc_ranks[subrank-1],tensor_dim[subrank],1])
        else:
            
            mat = tt_functions.unfolding_submatrix(tensor_entry, FI, FJ, subrank, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            uj = u[:,:trunc_ranks[subrank]]
            #if trunc_ranks[subrank] >= min(mat.shape):
            #    uj,_ = np.linalg.qr(mat)
            #else:
            #    uj, _, _ = sp.sparse.linalg.svds(mat, k=trunc_ranks[subrank])
            
            subcomm.send(uj,dest=subrank+1,tag=11)
            uj_tilde = np.reshape(uj,[len(FI[subrank-1]),tensor_dim[subrank]*trunc_ranks[subrank]])
            uj_prev = subcomm.recv(source=subrank-1,tag=11)
            A = np.linalg.lstsq(uj_prev[I[subrank-1],:],uj_tilde,rcond = None)[0]
            cores= np.reshape(A,[trunc_ranks[subrank-1],tensor_dim[subrank],trunc_ranks[subrank]])
            

        cores_all = subcomm.gather(cores,root=0)

        return cores_all

def ltr_nonnested_construction(tensor_entry,FIt,FJt,sample_col,sample_row,truncation_rank,tensor_dim):
   
    cores = []

    K_list = []
    L_list = []
    
    
    for i in range(len(tensor_dim)-1):
        L_list.append(samplingfunctions.tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
    for i in range(len(tensor_dim)-1):
        K_list.append(samplingfunctions.tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))

    uFJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
    uFI = [FIt[i] + K_list[i] for i in range(len(FIt))]
    
    for i in range(len(tensor_dim)):
        if i==0:
            B_test = tt_functions.unfolding_submatrix(tensor_entry,uFI,uFJ, i, tensor_dim, 1)

            q,r = sp.linalg.qr(B_test,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            cores.append(np.reshape(u[:,:truncation_rank[i]],[1,tensor_dim[i],truncation_rank[i]]))

        elif i==len(tensor_dim)-1:

            A_test = tt_functions.unfolding_submatrix(tensor_entry, uFI,uFJ, i-1, tensor_dim, 0)
            F = np.zeros((len(uFI[i-1]),truncation_rank[i-1]))
            for ir in range(len(uFI[i-1])):
                row  = cores[0][:,uFI[i-1][ir][0],:]
                for j in range(1,i):
                    row = row@cores[j][:,uFI[i-1][ir][j],:]
                F[ir,:] = row[0]
            
            B = np.linalg.lstsq(F,A_test,rcond = None)[0]
            cores.append(np.reshape(B,[truncation_rank[i-1],tensor_dim[i],1]))
        else:
            
            fst_ind = np.repeat(uFI[i-1],len(uFJ[i])*tensor_dim[i],axis = 0)
            mid_ind = np.tile(np.repeat(np.arange(tensor_dim[i]),len(uFJ[i])),len(uFI[i-1]))
            lst_ind = np.tile(uFJ[i],(len(uFI[i-1])*tensor_dim[i],1))
            ind = np.column_stack([fst_ind,mid_ind,lst_ind])
            t = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[i-1]),tensor_dim[i]*len(uFJ[i])])

            #t = np.reshape(tensor_entry(full_inds,tensor_dim),[len(I_test[i-1]),tensor_dim[i]*len(J_test[i])])
            #print(np.linalg.norm(t - t_test))
            F = np.zeros((len(uFI[i-1]),truncation_rank[i-1]))
            for ir in range(len(uFI[i-1])):
                row  = cores[0][:,uFI[i-1][ir][0],:]
                for j in range(1,i):
                    row = row@cores[j][:,uFI[i-1][ir][j],:]
                F[ir,:] = row[0]
            
            
            B = np.reshape(np.linalg.lstsq(F,t,rcond = None)[0],[truncation_rank[i-1]*tensor_dim[i],len(uFJ[i])])

            q,r = sp.linalg.qr(B,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            u = q@U
            cores.append(np.reshape(u[:,:truncation_rank[i]],[truncation_rank[i-1],tensor_dim[i],truncation_rank[i]]))
    
    return cores


def rtl_nonnested_construction(tensor_entry,FIt,FJt,sample_col,sample_row,trunc_ranks,tensor_dim):


    K_list = []
    L_list = []

    for i in range(len(tensor_dim)-1):
        L_list.append(samplingfunctions.tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
    for i in range(len(tensor_dim)-1):
        K_list.append(samplingfunctions.tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))
 
    uFJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
    uFI = [FIt[i] + K_list[i] for i in range(len(FIt))]

    core_rev_t = []
    for i in range(len(tensor_dim)-1,-1,-1):
        if i == len(tensor_dim)-1:
            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
            Q,r = sp.linalg.qr(mat,mode = "economic")
            U,_,_ = np.linalg.svd(r)
            q = Q@U
            
            core_rev_t.append(np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],1]))
        else:
            F = np.zeros((trunc_ranks[i],len(uFJ[i])))
            for j in range(len(uFJ[i])):
                ind = list(uFJ[i][j])
                col = core_rev_t[0][:,ind[0],:]
                for l in range(1,len(core_rev_t)):
                    col = col@core_rev_t[l][:,ind[l],:]
                F[:,[j]] = col
            
            mat = tt_functions.unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            y = np.linalg.lstsq(F.T,mat.T,rcond = None)[0].T
            
            if i>0:
                y= np.reshape(y,[len(uFI[i-1]),tensor_dim[i]*trunc_ranks[i]]).T
                Q,r = sp.linalg.qr(y,mode = "economic")
                U,_,_ = np.linalg.svd(r)
                q = Q@U
                
                core_rev_t.insert(0,np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))
            else:
                core_rev_t.insert(0,np.reshape(y,[1,tensor_dim[0],trunc_ranks[0]]))
    return core_rev_t




