import sys
import numpy as np
import tt_functions
import core_construct


# Options: "Hilbert", "Matern", "ThinPlateSpline", "2d2v", "3d3v".
tensor_def = "3d3v"


def tensor_entry(indices,dim):

    inds = np.asarray(indices)

    if tensor_def == "Matern":
        index_tuple = np.array([np.cos((inds[:,i]+1/2)*np.pi/(dim[i])) for i in range(len(dim))]).T
        index_x = ((3-2)/2)*index_tuple[:,:len(dim)//2] + ((3+2)/2)
        index_y = ((1-0)/2)*index_tuple[:,len(dim)//2:] + ((1+0)/2)
        norm = np.linalg.norm(index_x - index_y,axis = 1)
        val = (1.0 + np.sqrt(5.0)*norm + (5/3)*norm**2) * np.exp(-np.sqrt(5)*norm)

    elif tensor_def == "ThinPlateSpline":
        index_tuple = np.array([np.cos((inds[:,i]+1/2)*np.pi/(dim[i])) for i in range(len(dim))]).T
        index_x = ((3-2)/2)*index_tuple[:,:len(dim)//2] + ((3+2)/2)
        index_y = ((1-0)/2)*index_tuple[:,len(dim)//2:] + ((1+0)/2)
        norm = np.linalg.norm(index_x - index_y,axis = 1)
        val = np.log(norm**2)*norm**2

    elif tensor_def == "2d2v":
        ax,ay,bx,by = -1/2,-1/2,1/2,1/2
        avx,avy,bvx,bvy = -3,-3,3,3
        x = ax + (inds[:,0]/(dim[0]-1))*(bx-ax)
        y = ay + (inds[:,2]/(dim[2]-1))*(by-ay)
        vx = avx + (inds[:,1]/(dim[1]-1))*(bvx - avx)
        vy = avy + (inds[:,3]/(dim[3]-1))*(bvy - avy)
        rhox = 1+(7/8)*np.sin(2*np.pi*x)
        Tx = (1/2)+(2/5)*np.sin(2*np.pi*x)
        rhoy = 1+(7/8)*np.sin(2*np.pi*y)
        Ty = (1/2)+(2/5)*np.sin(2*np.pi*y)
        val = (((rhox)/(2*np.sqrt(2*np.pi*(Tx))))+((rhoy)/(2*np.sqrt(2*np.pi*Ty))))*(np.exp(-((np.abs(vx - 3/4)**2)/(2*Tx))-((np.abs(vy - 3/4)**2)/(2*Ty))) + np.exp(-((np.abs(vx + 3/4)**2)/(2*Tx))-((np.abs(vy + 3/4)**2)/(2*Ty))) )

    elif tensor_def == "3d3v":
        ax,ay,az,bx,by,bz = -1/2,-1/2,-1/2,1/2,1/2,1/2
        avx,avy,avz,bvx,bvy,bvz = -3,-3,-3,3,3,3
        x = ax + (inds[:,0]/(dim[0]-1))*(bx-ax)
        y = ay + (inds[:,2]/(dim[2]-1))*(by-ay)
        z = az + (inds[:,4]/(dim[4]-1))*(bz-az)
        vx = avx + (inds[:,1]/(dim[1]-1))*(bvx - avx)
        vy = avy + (inds[:,3]/(dim[3]-1))*(bvy - avy)
        vz = avz + (inds[:,5]/(dim[5]-1))*(bvz - avz)
        rhox = 1+(7/8)*np.sin(2*np.pi*x)
        Tx = (1/2)+(2/5)*np.sin(2*np.pi*(x))
        rhoy = 1+(7/8)*np.sin(2*np.pi*y)
        Ty = (1/2)+(2/5)*np.sin(2*np.pi*(y))
        rhoz = 1+(7/8)*np.sin(2*np.pi*z)
        Tz = (1/2)+(2/5)*np.sin(2*np.pi*(z))
        val = 1+(((rhox)/(2*np.sqrt((2*np.pi*(Tx))**3)))+((rhoy)/(2*np.sqrt((2*np.pi*Ty)**3)))+((rhoz)/(2*np.sqrt((2*np.pi*Tz)**3))))*(np.exp(-((np.abs(vx - 3/4)**2)/(2*Tx))-((np.abs(vy - 3/4)**2)/(2*Ty))-((np.abs(vz - 3/4)**2)/(2*Tz))) + np.exp(-((np.abs(vx + 3/4)**2)/(2*Tx))-((np.abs(vy + 3/4)**2)/(2*Ty))-((np.abs(vz + 3/4)**2)/(2*Tz))) )

    elif tensor_def == "Hilbert":
        val = 1/(np.sum(inds,axis = 1)+1)

    else:
        raise ValueError(f"Unknown tensor_def: {tensor_def}")

    return val


if __name__ == "__main__":

    # Hilbert works in any dimension; Matern and ThinPlateSpline need even dimensions.
    # 2d2v requires 4 dimensions, and 3d3v requires 6.
    dimension = 6 if tensor_def == "3d3v" else 4
    tensor_dim = [500 for _ in range(dimension)]
    if tensor_def not in ("Hilbert","Matern","ThinPlateSpline","2d2v","3d3v"):
        raise ValueError(f"Unknown tensor_def: {tensor_def}")
    if tensor_def in ("Matern","ThinPlateSpline") and len(tensor_dim)%2:
        raise ValueError("Matern and ThinPlateSpline require an even number of dimensions")
    if tensor_def == "2d2v" and len(tensor_dim)!=4:
        raise ValueError("2d2v requires 4 dimensions")
    if tensor_def == "3d3v" and len(tensor_dim)!=6:
        raise ValueError("3d3v requires 6 dimensions")
    sample_row = [30 for _ in range(len(tensor_dim)-1)]
    sample_col = [30 for _ in range(len(tensor_dim)-1)]
    min_rank = [2 for _ in range(len(tensor_dim)-1)]
    max_rank = [30 for _ in range(len(tensor_dim)-1)]
    tol = [1e-6 for _ in range(len(tensor_dim)-1)]
    adap_toler = None

    # Options for construction_type:
    # None    - TTACA without oversampling
    # "Par"   - parallel-form PEID, LTR
    # "Seq"   - sequential PEID, LTR
    # "Par2"  - two-sided nested PEID
    # "Seq2"  - two-sided nonnested PEID
    # "R-Par" - average LTR/RTL parallel-form PEID and round
    # "R-Seq" - average LTR/RTL sequential PEID and round
    # MPI modes "Par-P" and "Par2-P" require collective MPI calls and
    # at least len(tensor_dim) ranks; they are not shown in this serial example.
    
    construction_type = "Par"
    run_tntorch = True

    # PEID can be called directly in the ttaca function.
    cores_0,_,_ = tt_functions.ttaca_peid(tensor_entry,tensor_dim,sample_row,sample_col,min_rank,max_rank,tol,sampling_type = construction_type,adap_toler = adap_toler)

    # We can also call PEID constructors directly if we already have the indices from a previous run.
    cores_none,FI,FJ = tt_functions.ttaca_peid(tensor_entry,tensor_dim,sample_row,sample_col,min_rank,max_rank,tol,sampling_type = None)
    if construction_type is None:
        cores_1 = core_construct.ttaca_cores(tensor_entry,FI,FJ,[len(i) for i in FI],tensor_dim,{})
    elif construction_type == "Par":
        cores_1 = core_construct.peid_par(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler,orientation = "ltr")
    elif construction_type == "Seq":
        cores_1 = core_construct.peid_seq(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler,orientation = "ltr")
    elif construction_type == "Par2":
        cores_1 = core_construct.peid_par2(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
    elif construction_type == "Seq2":
        cores_1 = core_construct.peid_seq2(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
    elif construction_type == "R-Par":
        cores_1 = core_construct.peid_r(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,base = "par")
    elif construction_type == "R-Seq":
        cores_1 = core_construct.peid_r(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,base = "seq")

    error = tt_functions.sampled_error(tensor_entry,tensor_dim,1000,cores_0,cores_none,cores_1)
    print("*"*60)
    print("TTACA example")
    print("Tensor:                          ",tensor_def)
    print("Construction type:               ",construction_type)
    print("Error through ttaca_peid:        ",f"{error[0]:.3e}")
    print("Error of base TTACA cores:       ",f"{error[1]:.3e}")
    print("Error through direct constructor:",f"{error[2]:.3e}")
    print("*"*60,"\n")

    if run_tntorch:
        try:
            import torch as tn
            import tntorch as tntt
        except ModuleNotFoundError as exc:
            print("Skipping the TnTorch example; missing dependency:",exc.name)
            sys.exit(0)

        tn.set_default_dtype(tn.float64)

        # TnTorch cross function
        max_rank_tntorch = [len(i) for i in FI]
        t,info = tntt.cross(function = lambda indices: tn.from_numpy(tensor_entry(indices.detach().cpu().numpy(),tensor_dim)).to(indices),function_arg = "matrix",ranks_tt = max_rank_tntorch,domain = [tn.arange(n,dtype = tn.float64) for n in tensor_dim],verbose = False,return_info = True)

        # Convert the TnTorch cores for the NumPy error evaluator.
        torch_cores = [core.detach().cpu().numpy() for core in t.cores]

        # Convert TnTorch indices, then call the selected PEID constructor directly.
        FI,FJ = core_construct.torch_ind_extract(info)
        if construction_type is None:
            cores_torch = torch_cores
        elif construction_type == "Par":
            cores_torch = core_construct.peid_par(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler,orientation = "ltr")
        elif construction_type == "Seq":
            cores_torch = core_construct.peid_seq(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler,orientation = "ltr")
        elif construction_type == "Par2":
            cores_torch = core_construct.peid_par2(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
        elif construction_type == "Seq2":
            cores_torch = core_construct.peid_seq2(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = adap_toler)
        elif construction_type == "R-Par":
            cores_torch = core_construct.peid_r(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,base = "par")
        elif construction_type == "R-Seq":
            cores_torch = core_construct.peid_r(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,base = "seq")

        error = tt_functions.sampled_error(tensor_entry,tensor_dim,1000,torch_cores,cores_torch)
        print("*"*60)
        print("TnTorch example")
        print("Tensor:                    ",tensor_def)
        print("Ranks from TnTorch indices:",[len(i) for i in FI])
        print("Construction type:         ",construction_type)
        print("Error of TnTorch cores:    ",f"{error[0]:.3e}")
        print("Error of PEID cores:       ",f"{error[1]:.3e}")
        print("*"*60,"\n")
