import h5py

f = h5py.File("2017-05-12_batchdata_updated_struct_errorcorrect.mat", "r")
print(list(f.keys()))
