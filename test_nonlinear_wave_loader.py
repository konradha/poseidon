from scOT.problems.wave.nonlinear import NonlinearWave
import torch
from sys import argv

if __name__ == '__main__':
    test_dataset = NonlinearWave(
        data_path=str(argv[1]), 
        which="train",
        num_trajectories=10
    )

    item = test_dataset[0]
    print(f"Input shape: {item['pixel_values'].shape}")
    print(f"Label shape: {item['labels'].shape}")

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True
    )
    batch = next(iter(dataloader))
    print(f"Batch input shape: {batch['pixel_values'].shape}")
    print(f"Batch label shape: {batch['labels'].shape}")
