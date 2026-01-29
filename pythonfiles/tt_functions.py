import math
import numpy as np
import indexingfunctions
import samplingfunctions
import osfunctions
import random
import sys
import time
try:
    from mpi4py import MPI
except Exception:
    MPI = None

def sampled_error(tensor_entry,dim,samples,*args):
    np.random.seed(10)
    error = np.zeros(len(args))
    norms = np.zeros(len(args))
    for _ in range(samples):
        index = tuple([random.randint(0,dim[j]-1) for j in range(len(dim))])
        a = tensor_entry([index],dim)[0]
        for i in range(len(args)):
            b = Core_to_Tensor_Value(args[i],tuple(index))   
            error[i] += (a-b)**2
            norms[i] += a**2 
    np.random.seed(None)
    return [np.sqrt(error[i])/np.sqrt(norms[i]) for i in range(len(args))]

"""
def unfolding_submatrix(tensor_entry,uFI,uFJ,k,tensor_dim,axis):
    if axis == 0:
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

    elif axis == 1:
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
       
    return submatrix
"""

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

def Core_to_Tensor_Value(cores,index):
   
    start = cores[0][:,index[0],:]
    for i in range(1,len(cores)):
        start = start@cores[i][:,index[i],:]
    value = start[0][0]
    return value


def tt_add(core1,core2,tensor_dim,trunc_ranks):
   
    cores = []
    cores.append(np.concatenate((core1[0],core2[0]),axis = 2))
   
    for i in range(1,len(core1)-1):
        cores.append(np.zeros((core1[i].shape[0]+core2[i].shape[0],tensor_dim[i],core1[i].shape[2]+core2[i].shape[2])))
        cores[i][:trunc_ranks[i-1],:,:trunc_ranks[i]] = core1[i]
        cores[i][trunc_ranks[i-1]:,:,trunc_ranks[i]:] = core2[i]
   
       
    cores.append((2**(-1))*np.concatenate((core1[-1],core2[-1]),axis = 0))
   
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


"""
def greedy_search(tensor_entry,I,J,FI,FJ,tensor_dim,k,tol=0,sample_size = 100,maxiter = 1):
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
            #unavail_row = [x for y in FI[0] for x in y]
            #unavail_col = [indexingfunctions.Large_ravel(i,[tensor_dim[k+1],len(J[k+1])]) for i in J[k]]
            
            s_row = samplingfunctions.cheap_sample(0,tensor_dim[k],unavail_row,sample_size)
            s_col = samplingfunctions.cheap_sample(0,tensor_dim[k+1]*len(J[k+1]),unavail_col,sample_size)
            t_row = [i for i in s_row]
            t_col = [tuple(reversed(indexingfunctions.Large_unravel(i,[len(FJ[k+1]),tensor_dim[k+1]]))) for i in s_col]
            samples = [[t_row[i]]+[t_col[i][0]] + FJ[k+1][t_col[i][1]] for i in range(min(len(t_row),len(t_col)))]
        elif k==len(tensor_dim)-2:
            #unavail_row = [indexingfunctions.Large_ravel(i,[len(I[k-1]),tensor_dim[k]]) for i in I[k]]
            #unavail_col = [x for y in J[k] for x in y]
            s_row = samplingfunctions.cheap_sample(0,len(I[k-1])*tensor_dim[k],unavail_row,sample_size)
            s_col = samplingfunctions.cheap_sample(0,tensor_dim[k+1],unavail_col,sample_size)
            t_row = [indexingfunctions.Large_unravel(i,[len(FI[k-1]),tensor_dim[k]]) for i in s_row]
            t_col = [i for i in s_col]
            
            samples = [FI[k-1][t_row[i][0]]+[t_row[i][1]]+[t_col[i]] for i in range(min(len(t_row),len(t_col)))]
        else:
            
            #unavail_row = [indexingfunctions.Large_ravel(i,[len(I[k-1]),tensor_dim[k]]) for i in I[k]]
            #unavail_col = [indexingfunctions.Large_ravel(i,[tensor_dim[k+1],len(J[k+1])]) for i in J[k]]
            s_row = samplingfunctions.cheap_sample(0,len(I[k-1])*tensor_dim[k],unavail_row,sample_size)
            s_col = samplingfunctions.cheap_sample(0,tensor_dim[k+1]*len(J[k+1]),unavail_col,sample_size)
            t_row = [indexingfunctions.Large_unravel(i,[len(FI[k-1]),tensor_dim[k]]) for i in s_row]
            t_col = [tuple(reversed(indexingfunctions.Large_unravel(i,[len(FJ[k+1]),tensor_dim[k+1]]))) for i in s_col]
            
            samples = [FI[k-1][t_row[i][0]]+[t_row[i][1]]+[t_col[i][0]] + FJ[k+1][t_col[i][1]] for i in range(min(len(t_row),len(t_col)))]
        

        vals = tensor_entry(samples,tensor_dim)

        #Selection
        index = np.argmax(np.abs(vals))
        i_star = np.array(samples[index][:k+1])
        j_star = np.array(samples[index][k+1:])
        p = vals[index]

        #loop
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
                        i_star_super = indexingfunctions.Large_ravel([i,i_star[-1]],[len(I[k-1]),tensor_dim[k]])
                        break
            
            row_val = tensor_entry(ind,tensor_dim) - Ej[i_star_super,:]@np.diag(D)@Ei
            
            #if k==0:
            #    print(l,"HERE ARE ROW VALS",row_val)
            #inds = np.argpartition(np.abs(row_val),len(FJ[k])+1)[-len(FJ[k])-1:]
            
            #sorted_inds = np.flip(inds[np.argsort(row_val[inds])])
            sorted_inds = np.flip(np.argsort(np.abs(row_val)))
            for j in sorted_inds:
                if j not in unavail_col:
                    p = row_val[j]
                    break
            
            #if k==0:
            #    print("This was the selection",j,p)
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
                        j_star_super = indexingfunctions.Large_ravel([i,j_star[0]],[len(J[k+1]),tensor_dim[k+1]])
                        break

            col_val = tensor_entry(ind,tensor_dim) - Ej@np.diag(D)@Ei[:,j_star_super]
            
            #inds = np.argpartition(np.abs(col_val),len(FI[k])+1)[-len(FI[k])-1:]
            
            #sorted_inds = np.flip(inds[np.argsort(col_val[inds])])
            
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
                    i_star_super = indexingfunctions.Large_ravel([i,i_star[-1]],[len(I[k-1]),tensor_dim[k]])

        if k==len(tensor_dim)-2:
            j_star_super = j_star[0]
        else:
            for i in range(len(FJ[k+1])):
                if list(j_star[1:])==FJ[k+1][i]:
                    j_star_super = indexingfunctions.Large_ravel([i,j_star[0]],[len(J[k+1]),tensor_dim[k+1]])
                    
        
        return list(i_star),list(j_star),i_star_super,j_star_super,p
"""

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
            s_row = samplingfunctions.cheap_sample(0, tensor_dim[k], unavail_row, sample_size)
            s_col = samplingfunctions.cheap_sample(0, tensor_dim[k + 1] * len(J[k + 1]), unavail_col, sample_size)
            t_row = [i for i in s_row]
            t_col = [tuple(reversed(indexingfunctions.Large_unravel(i, [len(FJ[k + 1]), tensor_dim[k + 1]]))) for i in s_col]
            samples = [[t_row[i]]+[t_col[i][0]] + FJ[k+1][t_col[i][1]] for i in range(min(len(t_row),len(t_col)))]
        elif k==len(tensor_dim)-2:
            s_row = samplingfunctions.cheap_sample(0, len(I[k - 1]) * tensor_dim[k], unavail_row, sample_size)
            s_col = samplingfunctions.cheap_sample(0, tensor_dim[k + 1], unavail_col, sample_size)
            t_row = [indexingfunctions.Large_unravel(i, [len(FI[k - 1]), tensor_dim[k]]) for i in s_row]
            t_col = [i for i in s_col]
            
            samples = [FI[k-1][t_row[i][0]]+[t_row[i][1]]+[t_col[i]] for i in range(min(len(t_row),len(t_col)))]
        else:
            s_row = samplingfunctions.cheap_sample(0, len(I[k - 1]) * tensor_dim[k], unavail_row, sample_size)
            s_col = samplingfunctions.cheap_sample(0, tensor_dim[k + 1] * len(J[k + 1]), unavail_col, sample_size)
            t_row = [indexingfunctions.Large_unravel(i, [len(FI[k - 1]), tensor_dim[k]]) for i in s_row]
            t_col = [tuple(reversed(indexingfunctions.Large_unravel(i, [len(FJ[k + 1]), tensor_dim[k + 1]]))) for i in s_col]
            
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
                        i_star_super = indexingfunctions.Large_ravel([i, i_star[-1]], [len(I[k - 1]), tensor_dim[k]])
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
                        j_star_super = indexingfunctions.Large_ravel([i, j_star[0]], [len(J[k + 1]), tensor_dim[k + 1]])
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
                    i_star_super = indexingfunctions.Large_ravel([i, i_star[-1]], [len(I[k - 1]), tensor_dim[k]])

        if k==len(tensor_dim)-2:
            j_star_super = j_star[0]
        else:
            for i in range(len(FJ[k+1])):
                if list(j_star[1:])==FJ[k+1][i]:
                    j_star_super = indexingfunctions.Large_ravel([i, j_star[0]], [len(J[k + 1]), tensor_dim[k + 1]])

        return list(i_star),list(j_star),i_star_super,j_star_super,p,Ej,D,Ei

def kickstart(tensor_entry,tensor_dim,maxiter = 10):

    ind = [random.randint(0,i-1) for i in tensor_dim]

    for i in range(maxiter):
        prev_ind = [i for i in ind]

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
            ind[j] = np.argmax(np.abs(vals))

        if prev_ind == ind:

            break
    FI = [[ind[:i+1]] for i in range(len(tensor_dim)-1)]
    FJ = [[ind[i+1:]] for i in range(len(tensor_dim)-1)]
    p = tensor_entry([ind],tensor_dim)[0]
    I = [[ind[i]] for i in range(len(tensor_dim)-1)]
    J = [[ind[i+1]] for i in range(len(tensor_dim)-1)]
    return I,J,FI,FJ,p


def kickstart_nonconvx(tensor_entry,tensor_dim,mpi=False,maxiter=10,num_samples = 10):
    if mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        local_num = np.ceil(num_samples/size)
        inds = [[random.randint(0,i-1) for i in tensor_dim] for _ in range(num_samples)]
        locs = []
        for ind in inds:
            for i in range(maxiter):
                prev_ind = [i for i in ind]

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
                    ind[j] = np.argmax(np.abs(vals))

                if prev_ind == ind:

                    break
            locs.append(ind)
        values = tensor_entry(locs,tensor_dim)
        ind = locs[np.argmax(np.abs(values))]
        
        ind = comm.gather(ind,root = 0)
        values = comm.gather(values,root = 0)
        if rank==0:
            
            ind = ind[np.argmax(np.abs(tensor_entry(ind,tensor_dim)))]
            FI = [[ind[:i+1]] for i in range(len(tensor_dim)-1)]
            FJ = [[ind[i+1:]] for i in range(len(tensor_dim)-1)]
            p = tensor_entry([ind],tensor_dim)[0]
            I = [[ind[i]] for i in range(len(tensor_dim)-1)]
            J = [[ind[i+1]] for i in range(len(tensor_dim)-1)]
            return I,J,FI,FJ,p
        else:
            return None,None,None,None,None

    else:

        inds = [[random.randint(0,i-1) for i in tensor_dim] for _ in range(num_samples)]
        locs = []
        for ind in inds:
            for i in range(maxiter):
                prev_ind = [i for i in ind]

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
                    ind[j] = np.argmax(np.abs(vals))

                if prev_ind == ind:

                    break
            locs.append(ind)
        values = tensor_entry(locs,tensor_dim)
        ind = locs[np.argmax(np.abs(values))]
        FI = [[ind[:i+1]] for i in range(len(tensor_dim)-1)]
        FJ = [[ind[i+1:]] for i in range(len(tensor_dim)-1)]
        p = tensor_entry([ind],tensor_dim)[0]
        I = [[ind[i]] for i in range(len(tensor_dim)-1)]
        J = [[ind[i+1]] for i in range(len(tensor_dim)-1)]
    return I,J,FI,FJ,p

def greedy_cross(tensor_entry,tensor_dim,min_rank = None,max_rank = None,tol = None,core_construction=True):

    t_start = time.time()
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
        

    I,J,FI,FJ,p = kickstart(tensor_entry,tensor_dim)

    tol = [np.abs(p)*i for i in tol]
    
    sweep = []
    for i in range(len(tensor_dim)-1):
        if max_rank[i]>1:
            sweep.append(i)
    count = [0 for _ in range(len(tensor_dim)-1)]
    pivots = []
    Ejs = [[] for _ in range(len(tensor_dim)-1)]
    Ds = [[] for _ in range(len(tensor_dim)-1)]
    Eis = [[] for _ in range(len(tensor_dim)-1)]
    for i in sweep:
        count[i]+=1

        #i_star,j_star,i_star_super,j_star_super,p = greedy_search(tensor_entry,I,J,FI,FJ,tensor_dim,i)
        i_star,j_star,i_star_super,j_star_super,p,Ejs[i],Ds[i],Eis[i] = greedy_search(tensor_entry,I,J,FI,FJ,Ejs[i],Ds[i],Eis[i],tensor_dim,i)

        I[i].append(i_star_super)
        J[i].append(j_star_super)
        FI[i].append(i_star)
        FJ[i].append(j_star)
        

        if count[i]<max_rank[i]-1 and np.abs(p)>tol[i]:
            
            sweep.append(i)

    FI = [[[int(i) for i in FI[l][j]] for j in range(len(FI[l]))] for l in range(len(tensor_dim)-1)]
    FJ = [[[int(i) for i in FJ[l][j]] for j in range(len(FJ[l]))] for l in range(len(tensor_dim)-1)]
    trunc_ranks = [len(i) for i in FI]
    
    print("Time for search",time.time() - t_start)
    if core_construction:
        print("HERE")
        cores = osfunctions.ltr_nested_construction(tensor_entry,FI,FJ,[0 for _ in range(len(tensor_dim)-1)],[0 for _ in range(len(tensor_dim)-1)],[len(i) for i in FI],tensor_dim,{})
        print("Now HERE")
        return cores,[FI,FJ,trunc_ranks]
    else:
        info = [I,J,FI,FJ,trunc_ranks]
        return info 
    


def tt_cross_os(tensor_entry,tensor_dim,sample_row,sample_col,min_rank = None,max_rank = None,tol = None,sampling_type = None):
    
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
        

    I,J,FI,FJ,p = kickstart(tensor_entry,tensor_dim)

    tol = [np.abs(p)*i for i in tol]
    
    sweep = []
    for i in range(len(tensor_dim)-1):
        if max_rank[i]>1:
            sweep.append(i)
    count = [0 for _ in range(len(tensor_dim)-1)]
    pivots = []
    Ejs = [[] for _ in range(len(tensor_dim)-1)]
    Ds = [[] for _ in range(len(tensor_dim)-1)]
    Eis = [[] for _ in range(len(tensor_dim)-1)]
    for i in sweep:
        count[i]+=1

        i_star,j_star,i_star_super,j_star_super,p,Ejs[i],Ds[i],Eis[i] = greedy_search(tensor_entry,I,J,FI,FJ,Ejs[i],Ds[i],Eis[i],tensor_dim,i)
        

        I[i].append(i_star_super)
        J[i].append(j_star_super)
        FI[i].append(i_star)
        FJ[i].append(j_star)
        

        if count[i]<max_rank[i]-1 and np.abs(p)>tol[i]:
            
            sweep.append(i)

    FI = [[[int(i) for i in FI[l][j]] for j in range(len(FI[l]))] for l in range(len(tensor_dim)-1)]
    FJ = [[[int(i) for i in FJ[l][j]] for j in range(len(FJ[l]))] for l in range(len(tensor_dim)-1)]
    trunc_ranks = [len(i) for i in FI]
    
    if sampling_type is None:
        cores = osfunctions.ltr_nested_construction(tensor_entry,FI,FJ,[0 for _ in range(len(tensor_dim)-1)],[0 for _ in range(len(tensor_dim)-1)],trunc_ranks,tensor_dim)
    elif sampling_type == "Par":
        cores = osfunctions.ltr_nested_construction(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim)
    elif sampling_type == "Par2":
        cores = osfunctions.two_sided_nested_constructionv2(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim)
    elif sampling_type == "Seq":
        cores = osfunctions.ltr_nonnested_construction(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim)
    elif sampling_type == "Seq2":
        cores = osfunctions.two_sided_nonnested_constructionv2(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim)
    elif sampling_type == "R":
        cores_left = osfunctions.ltr_nested_construction(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim)
        cores_right = osfunctions.rtl_nested_construction(tensor_entry,FI,FJ,sample_row,sample_col,trunc_ranks,tensor_dim)
        cores_add = tt_add(cores_left,cores_right,tensor_dim,trunc_ranks)
        cores = tt_round(cores_add,0,trunc_ranks)
    return cores

def tt_cross_parallel(tensor_entry,tensor_dim,sample_row,sample_col,min_rank = None,max_rank = None,tol = None,sampling_type = None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank < len(tensor_dim)-1:
        color = 1
        key = rank
    elif rank >= len(tensor_dim)-1 and rank < len(tensor_dim):
        color = 2
        key = rank
    else:
        color = 3
        key = rank

    subcomm = comm.Split(color,key)
    subrank = subcomm.Get_rank()
    subsize = subcomm.Get_size()

    if color == 1:
        if subsize < len(tensor_dim)-1:
            raise ValueError("Not enough mpi ranks to run. Must give ", len(tensor_dim)-1, "ranks")
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
            
        if subrank==0:
            #Run kickstart on rank 0 only and inform all other ranks of starting point
            I,J,FI,FJ,p = kickstart(tensor_entry,tensor_dim)
        else:
            I,J,FI,FJ,p = None,None,None,None,None

        I = subcomm.bcast(I,root = 0)
        J = subcomm.bcast(J,root = 0)
        FI = subcomm.bcast(FI,root = 0)
        FJ = subcomm.bcast(FJ,root = 0)
        p = subcomm.bcast(p,root = 0)

        tol = [np.abs(p)*i for i in tol]

        active = False

        if max_rank[subrank]>1:
            active = True

        count = 0
        Ej = []
        D = []
        Ei = []
        for i in range(max_rank[subrank]-1):
            if active:
                
                i_star,j_star,i_star_super,j_star_super,p,Ej,D,Ei = greedy_search(tensor_entry,I,J,FI,FJ,Ej,D,Ei,tensor_dim,rank)
                I[subrank].append(i_star_super)
                J[subrank].append(j_star_super)
                FI[subrank].append(i_star)
                FJ[subrank].append(j_star)
                if np.abs(p)<tol[subrank]:
                    active = False
                
            else:
                i_star,j_star,i_star_super,j_star_super = None,None,None,None
            
        
            if subrank==0:
                
                subcomm.send([i_star,i_star_super],subrank+1,tag = 0)
                inds_j = subcomm.recv(source = subrank+1,tag = 1)
                if inds_j[0] is not None:
                    FJ[subrank+1].append(inds_j[0])
                    J[subrank+1].append(inds_j[1])
                

            elif subrank == len(tensor_dim)-2:
                
                subcomm.send([j_star,j_star_super],subrank-1,tag = 1)
                inds_i = subcomm.recv(source = subrank-1,tag = 0)
                if inds_i[0] is not None:
                    FI[subrank-1].append(inds_i[0])
                    I[subrank-1].append(inds_i[1])
                
            else:
                
                subcomm.send([i_star,i_star_super],subrank+1,tag = 0)
                subcomm.send([j_star,j_star_super],subrank-1,tag = 1)
                inds_i = subcomm.recv(source = subrank-1,tag = 0)
                inds_j = subcomm.recv(source = subrank+1,tag = 1)

                if inds_i[0] is not None:
                    FI[subrank-1].append(inds_i[0])
                    I[subrank-1].append(inds_i[1])
                if inds_j[0] is not None:
                    FJ[subrank+1].append(inds_j[0])
                    J[subrank+1].append(inds_j[1])
                
        

            if subrank !=0:
                subcomm.send([FI[subrank],FJ[subrank]],dest = 0,tag = 0)
            else:
                info = [[FI[0],FJ[0]]]
                for i in range(1,len(tensor_dim)-1):
                    info.append(subcomm.recv(source = i,tag = 0))
        
        if subrank==0:
            FI_global = [[[int(i) for i in info[l][0][j]] for j in range(len(info[l][0]))] for l in range(len(tensor_dim)-1)]
            FJ_global = [[[int(i) for i in info[l][1][j]] for j in range(len(info[l][1]))] for l in range(len(tensor_dim)-1)]
            
        else:
            FI_global = None
            FJ_global = None

    if color!=1:
        FI_global = None
        FJ_global = None
    

    FI_global = comm.bcast(FI_global,root = 0)
    FJ_global = comm.bcast(FJ_global,root = 0)
    trunc_ranks = [len(i) for i in FI_global]
    #print(trunc_ranks,flush=True)
    #print(rank,"Made it here",flush=True)
    cores = osfunctions.ltr_nested_construction_parallel(tensor_entry, FI_global, FJ_global, sample_row, sample_col, trunc_ranks, tensor_dim)
    return cores



    
