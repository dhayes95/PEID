import numpy as np
import tt_functions
import indexingfunctions
import osfunctions
import torch as tn
import tntorch as tntt


def tensor_entry(indices,dim):

    if type(indices[0][0]) == tn.Tensor:
        inds = np.column_stack(indices).T
        value = tn.from_numpy(1/(np.sum(inds,axis = 1)+1)).float()
    else:
        value = 1/(np.sum(indices,axis = 1)+1)
    
    return value


if __name__ == "__main__":


    tensor_dim = [500 for _ in range(4)]
    sample_row = [30 for _ in range(len(tensor_dim)-1)]
    sample_col = [30 for _ in range(len(tensor_dim)-1)]
    min_rank =   [2 for _ in range(len(tensor_dim)-1)]
    max_rank =   [30 for _ in range(len(tensor_dim)-1)]
    tol =        [1e-6 for _ in range(len(tensor_dim)-1)]


    """
    Options for construction_type: None/empty   - no oversampling, just base TT-Cross (3.1)
                                   Nested       - Algorithm 3.1
                                   Nonnested    - Algorithm 3.2
                                   TwoNested    - Algorithm 3.3
                                   TwoNonnested - Algorithm 3.4
                                   Average      - Algorithm 3.5
    """
    construction_type = "Nested"

    cores_none =tt_functions.tt_cross_os(tensor_entry,tensor_dim,sample_row,sample_col,min_rank,max_rank,tol) 
    cores = tt_functions.tt_cross_os(tensor_entry,tensor_dim,sample_row,sample_col,min_rank,max_rank,tol,construction_type)
    error = tt_functions.sampled_error(tensor_entry,tensor_dim,1000,cores_none,cores)
    print("*"*60)
    print("Errors for no oversampling:",error[0])
    print("Error for oversampling    :",error[1])
    print("*"*60,"\n")
    
    #Run TnTorch cross function
    t,info = tntt.cross(function = lambda i0,i1,i2,i3: tensor_entry([(i0,i1,i2,i3)],tensor_dim),ranks_tt = max_rank,domain = [tn.arange(tensor_dim[i]) for i in range(len(tensor_dim))],return_info=True)
    
    #Run an index conversion of the output info from tntorch and compute the actual rank observed
    I,J = indexingfunctions.torch_ind_extract(info)
    actual_rank = [len(i) for i in I]

    #Call core construction explicitly from osfunctions
    cores_torch = osfunctions.ltr_nested_construction(tensor_entry,I,J,sample_row,sample_col,actual_rank,tensor_dim)

