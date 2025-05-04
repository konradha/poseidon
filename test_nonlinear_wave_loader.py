from scOT.problems.wave.nonlinear import NonlinearWave
import torch
from sys import argv
from scOT.problems.base import get_dataset

if __name__ == '__main__':
    data_path = str(argv[1])
   
    print("direct:")
    test_dataset = NonlinearWave(
        data_path=data_path, 
        which="train",
        num_trajectories=10,
        max_num_time_steps=10,
        time_step_size=2,
    )

    item = test_dataset[0]
    print(f"Input shape: {item['pixel_values'].shape}")
    print(f"Label shape: {item['labels'].shape}")

    
    print("\nframework:")
    try:
        framework_dataset = get_dataset(
            dataset="wave.nonlinear",
            data_path=data_path,
            which="train",
            num_trajectories=10
        )
        print("✓ Dataset registered correctly in framework")
        
        framework_item = framework_dataset[0]
        print(f"Input shape: {framework_item['pixel_values'].shape}")
        print(f"Label shape: {framework_item['labels'].shape}")
        
    except Exception as e:
        print(f"✗ Framework registration failed: {str(e)}")
