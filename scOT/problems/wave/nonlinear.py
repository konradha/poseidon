import torch
import os
import h5py
import numpy as np
from glob import glob
from scOT.problems.base import BaseDataset

class NonlinearWave(BaseDataset):
    def __init__(
        self, 
        *args, 
        data_path=None,
        max_snapshots=None,
        normalize=True,
        **kwargs
    ):
        assert data_path is not None
        super().__init__(*args, data_path=data_path, **kwargs)

       
        self.file_paths = sorted(glob(os.path.join(self.data_path, "run_*.h5")))
        self.total_files = len(self.file_paths)
        
        self.N_max = self.total_files
        self.N_val = max(int(self.total_files * 0.1), 10)
        self.N_test = max(int(self.total_files * 0.1), 10) 
        
        with h5py.File(self.file_paths[0], 'r') as f:
            grid = f['grid']
            self.resolution = grid.attrs['nx']
    
            self.max_snapshots = max_snapshots or f['time'].attrs['num_snapshots']
            
            u_shape = f['u'].shape
            
        self.normalize = normalize
        
        self.input_dim = 2
           
        self.label_description = "[u,v]"
        self.output_dim = 2
       
        if normalize and not hasattr(self, 'constants'):
            self.compute_normalization_constants()
            
        self.post_init()
        
    def compute_normalization_constants(self):
        sample_size = min(100, self.total_files)
        sample_indices = np.random.choice(self.total_files, sample_size, replace=False)
        
        u_values = []
        v_values = []
        
        for idx in sample_indices:
            with h5py.File(self.file_paths[idx], 'r') as f:
                u_data = f['u'][0]
                v_data = f['v'][0] 
                
                u_values.append(u_data.flatten())
                v_values.append(v_data.flatten())
               
        u_all = np.concatenate(u_values)
        v_all = np.concatenate(v_values)
    
        self.constants = {
            "mean_u": float(np.mean(u_all)),
            "std_u": float(np.std(u_all)),
            "mean_v": float(np.mean(v_all)),
            "std_v": float(np.std(v_all)),
        }



    def __getitem__(self, idx):
        file_idx = idx + self.start
        
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            u0 = torch.from_numpy(f['initial_condition/u0'][:]).float()
            v0 = torch.from_numpy(f['initial_condition/v0'][:]).float()
            
            u_traj = torch.from_numpy(f['u'][:self.max_snapshots]).float()
            v_traj = torch.from_numpy(f['v'][:self.max_snapshots]).float()
                
            u0 = u0.reshape(1, self.resolution, self.resolution)
            v0 = v0.reshape(1, self.resolution, self.resolution)
            if self.normalize:
                u0 = (u0 - self.constants["mean_u"]) / self.constants["std_u"]
                v0 = (v0 - self.constants["mean_v"]) / self.constants["std_v"]
                
                u_traj = (u_traj - self.constants["mean_u"]) / self.constants["std_u"]
                v_traj = (v_traj - self.constants["mean_v"]) / self.constants["std_v"]
               
        inputs = torch.cat([u0, v0], dim=0)
            
        
        labels = torch.cat([
            u_traj[-1].unsqueeze(0),
            v_traj[-1].unsqueeze(0)
        ], dim=0)
        result = {
            "pixel_values": inputs,
            "labels": labels,
            "run_index": file_idx,
        }
       
        
        return result
