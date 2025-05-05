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
        normalize=False,
        normalization_type="minmax",
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

            time_group = f['time']
            self.total_time = float(time_group.attrs['T'])
            self.num_timesteps = int(time_group.attrs['nt'])
            self.num_snapshots = int(time_group.attrs['num_snapshots'])
            self.max_snapshots = self.num_snapshots
        
        self.normalize = normalize
        self.input_dim = 2
        self.label_description = "[u,v]"
        self.output_dim = 2
        
        if normalize and not hasattr(self, 'constants'):
            self.normalization_type = normalization_type
            self.compute_normalization_constants()
        
        if not hasattr(self, 'constants'):
            self.constants = {}
            
        self.constants["time_max"] = max_snapshots or nt
        self.post_init()
       
    def compute_normalization_constants(self):
        sample_size = min(350, self.total_files)
        sample_indices = np.random.choice(self.total_files, sample_size, replace=False)
        u_values = []
        v_values = []
        u_max_per_traj = []
        v_max_per_traj = []
        u_min_per_traj = []
        v_min_per_traj = []

        for idx in sample_indices:
            with h5py.File(self.file_paths[idx], 'r') as f:
                u_traj = f['u'][:self.max_snapshots]
                v_traj = f['v'][:self.max_snapshots]
                sample_points = min(1000, u_traj.size)
                flat_indices = np.random.choice(u_traj.size, sample_points, replace=False)
                u_sampled = u_traj.flatten()[flat_indices]
                v_sampled = v_traj.flatten()[flat_indices]
                u_values.append(u_sampled)
                v_values.append(v_sampled)
                u_max_per_traj.append(np.max(u_traj))
                v_max_per_traj.append(np.max(v_traj))
                u_min_per_traj.append(np.min(u_traj))
                v_min_per_traj.append(np.min(v_traj))

        u_all = np.concatenate(u_values)
        v_all = np.concatenate(v_values)
        self.constants = {
            "mean_u": float(np.mean(u_all)),
            "std_u": float(np.std(u_all)),
            "mean_v": float(np.mean(v_all)),
            "std_v": float(np.std(v_all)),
            "max_u": float(np.max(u_max_per_traj)),
            "min_u": float(np.min(u_min_per_traj)),
            "max_v": float(np.max(v_max_per_traj)),
            "min_v": float(np.min(v_min_per_traj)),
            "abs_max_u": float(max(abs(np.max(u_max_per_traj)), abs(np.min(u_min_per_traj)))),
            "abs_max_v": float(max(abs(np.max(v_max_per_traj)), abs(np.min(v_min_per_traj)))),
        }
        self.constants["u_range"] = self.constants["max_u"] - self.constants["min_u"]
        self.constants["v_range"] = self.constants["max_v"] - self.constants["min_v"]

    def normalize_data(self, data, key_prefix):
        if self.normalization_type == "standardize":
            return (data - self.constants[f"mean_{key_prefix}"]) / self.constants[f"std_{key_prefix}"]
        elif self.normalization_type == "minmax":
            return (data - self.constants[f"min_{key_prefix}"]) / self.constants[f"{key_prefix}_range"]
        elif self.normalization_type == "absmax":
            return data / self.constants[f"abs_max_{key_prefix}"]
        else:
            return data

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
                u0 = self.normalize_data(u0, "u")
                v0 = self.normalize_data(v0, "v")
                u_traj = self.normalize_data(u_traj, "u")
                v_traj = self.normalize_data(v_traj, "v")
        
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
        time_value = t2 / self.num_snapshots
        
        return {
            "pixel_values": input_state,
            "labels": labels,
            "time": time_value,
            "run_index": file_idx,
        }
