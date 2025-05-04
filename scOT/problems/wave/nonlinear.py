import torch
import os
import h5py
import numpy as np
from glob import glob
from scOT.problems.base import BaseTimeDataset

class NonlinearWave(BaseTimeDataset):
    def __init__(
        self, 
        *args, 
        data_path=None,
        max_snapshots=None,
        normalize=True,
        **kwargs
    ):
        super().__init__(*args, data_path=data_path, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20
        
        self.file_paths = sorted(glob(os.path.join(self.data_path, "*.h5")))
        self.total_files = len(self.file_paths)
        
        self.N_max = self.total_files
        val_ratio = min(0.1, 200/self.total_files)
        test_ratio = min(0.1, 200/self.total_files)
        
        self.N_val = max(int(self.total_files * val_ratio), 10)
        self.N_test = max(int(self.total_files * test_ratio), 10)
        
        available_for_train = self.N_max - self.N_val - self.N_test
        if available_for_train <= 0:
            self.N_val = int(self.N_max * 0.2)
            self.N_test = int(self.N_max * 0.2)
        
        with h5py.File(self.file_paths[0], 'r') as f:
            u_data = f['u']
            nt, nx, ny = u_data.shape
            self.resolution = nx
            self.max_snapshots = max_snapshots or nt
        
        self.normalize = normalize
        self.input_dim = 2
        self.label_description = "[u,v]"
        self.output_dim = 2
        
        if normalize and not hasattr(self, 'constants'):
            self.compute_normalization_constants()
        
        if not hasattr(self, 'constants'):
            self.constants = {}
            
        self.constants["time"] = float(self.max_snapshots)
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
        i, t, t1, t2 = self._idx_map(idx)
        file_idx = i + self.start
        
        with h5py.File(self.file_paths[file_idx], 'r') as f:            
            u_traj = torch.from_numpy(f['u'][:self.max_snapshots]).float()
            v_traj = torch.from_numpy(f['v'][:self.max_snapshots]).float()
            u0 = u_traj[0] 
            v0 = v_traj[0]
            
            u0 = u0.reshape(1, self.resolution, self.resolution)
            v0 = v0.reshape(1, self.resolution, self.resolution)
            
            if self.normalize:
                u0 = (u0 - self.constants["mean_u"]) / self.constants["std_u"]
                v0 = (v0 - self.constants["mean_v"]) / self.constants["std_v"]
                
                u_traj = (u_traj - self.constants["mean_u"]) / self.constants["std_u"]
                v_traj = (v_traj - self.constants["mean_v"]) / self.constants["std_v"]
        
        if t1 == 0:
            input_state = torch.cat([u0, v0], dim=0)
        else:
            t1_idx = min(t1, len(u_traj) - 1)
            input_state = torch.cat([
                u_traj[t1_idx].unsqueeze(0),
                v_traj[t1_idx].unsqueeze(0)
            ], dim=0)
        
        t2_idx = min(t2, len(u_traj) - 1)
        labels = torch.cat([
            u_traj[t2_idx].unsqueeze(0),
            v_traj[t2_idx].unsqueeze(0)
        ], dim=0)
        
        time_value = t / self.constants.get("time", self.max_snapshots)
        
        return {
            "pixel_values": input_state,
            "labels": labels,
            "time": time_value,
            "run_index": file_idx,
        }
