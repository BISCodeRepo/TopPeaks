import numpy as np
import h5py
import time
import progressbar

## read hdf5

start = time.time()

## Top model
## Binning model
isBin = False ## Binning model was deprecated since 2023. 10.
isTop = True
N = 30

with h5py.File("holdout_hcd_bin100.hdf5", 'r+') as data_set:
    
    size = len(data_set['intensities_raw'])
    
    print(str(size) + " were detected")
    
    int_array = np.array(data_set['intensities_raw'])
    mass_array = np.array(data_set['masses_raw'])
    
    ## sub sets
    indices = int_array.argsort()
    indices = indices[:,::-1]
    reset_indices = indices[:,N:]
    
    bar = progressbar.ProgressBar(maxval = size).start()
    
    for idx in range(size):
        bar.update(idx)
        if isTop:
            int_array[idx][reset_indices[idx]] = np.where(int_array[idx][reset_indices[idx]] > 0, 0, int_array[idx][reset_indices[idx]])
            mass_array[idx][reset_indices[idx]] = np.where(mass_array[idx][reset_indices[idx]] > 0, 0, mass_array[idx][reset_indices[idx]])
        
        if isBin:
            int_array[idx]= np.where(int_array[idx] > 0, np.round(int_array[idx], 2), int_array[idx])
    
    del data_set["intensities_raw"]
    data_set['intensities_raw'] = int_array
    del data_set["masses_raw"]
    data_set['masses_raw'] = mass_array
    
    
end = time.time()
print(f"{end - start:.5f} sec")


