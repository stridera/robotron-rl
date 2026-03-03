"""
Train CNN sprite detector from collected frames.

Architecture: Grid-based detection
- Input: 84x84 grayscale frame
- Output: Grid of sprite predictions (per-cell detection)
- Loss: Binary classification (sprite present?) + position regression + type classification

Simpler than full object detection (YOLO/Faster R-CNN) but sufficient for our use case.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import os


class SpriteDataset(Dataset):
    """Dataset for sprite detection."""

    def __init__(self, frames, sprite_data, grid_size=16):
        """
        Args:
            frames: (N, 84, 84) numpy array of grayscale frames
            sprite_data: List of N lists of (x, y, sprite_type) tuples
            grid_size: Number of grid cells (grid_size x grid_size)
        """
        self.frames = frames
        self.sprite_data = sprite_data
        self.grid_size = grid_size
        self.cell_size = 84 / grid_size  # pixels per cell

        # Sprite type encoding
        self.sprite_types = [
            'Player', 'Grunt', 'Electrode', 'Hulk', 'Sphereoid', 'Quark',
            'Brain', 'Enforcer', 'Tank', 'Mommy', 'Daddy', 'Mikey',
            'Prog', 'Cruise', 'Bullet', 'EnforcerBullet', 'TankShell'
        ]
        self.type_to_idx = {t: i for i, t in enumerate(self.sprite_types)}
        self.num_types = len(self.sprite_types)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        sprites = self.sprite_data[idx]

        # Convert frame to tensor: (1, 84, 84)
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)  # Add channel dim

        # Create grid targets
        # For each cell: [has_sprite (1), type_onehot (num_types), offset_x (1), offset_y (1)]
        target_size = 1 + self.num_types + 2
        targets = torch.zeros(self.grid_size, self.grid_size, target_size)

        # Fill in sprite information
        for x, y, sprite_type in sprites:
            # Convert sprite position to grid cell
            # Note: x, y are in play area coordinates (0-665, 0-492 from original)
            # But frames are 84x84, so we need to scale
            # Assume sprites are already scaled to 84x84 space by the game engine
            grid_x = int(x * self.grid_size / 84)
            grid_y = int(y * self.grid_size / 84)

            # Clamp to grid bounds
            grid_x = max(0, min(self.grid_size - 1, grid_x))
            grid_y = max(0, min(self.grid_size - 1, grid_y))

            # Calculate offset within cell (0 to 1)
            cell_x = x - (grid_x * self.cell_size)
            cell_y = y - (grid_y * self.cell_size)
            offset_x = cell_x / self.cell_size
            offset_y = cell_y / self.cell_size

            # Fill target
            targets[grid_y, grid_x, 0] = 1.0  # has_sprite

            # Type one-hot
            type_idx = self.type_to_idx.get(sprite_type, 0)
            targets[grid_y, grid_x, 1 + type_idx] = 1.0

            # Offsets
            targets[grid_y, grid_x, 1 + self.num_types] = offset_x
            targets[grid_y, grid_x, 1 + self.num_types + 1] = offset_y

        return frame_tensor, targets


class SpriteDetector(nn.Module):
    """
    Grid-based sprite detector using CNN backbone.

    Architecture:
    - Conv layers to extract features
    - Output grid with per-cell predictions
    """

    def __init__(self, grid_size=16, num_types=17):
        super().__init__()
        self.grid_size = grid_size
        self.num_types = num_types

        # CNN backbone: 84x84 -> features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 42x42
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 21x21
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 21x21

        # Upsample back to grid size
        # 21x21 -> 16x16 (adaptive pool)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Prediction head per cell
        # Output: has_sprite (1) + type (num_types) + offset (2)
        output_channels = 1 + num_types + 2
        self.pred_head = nn.Conv2d(256, output_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, 1, 84, 84)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # Pool to grid size
        x = self.pool(x)  # (batch, 256, grid_size, grid_size)

        # Predictions
        x = self.pred_head(x)  # (batch, output_channels, grid_size, grid_size)

        # Reshape to (batch, grid_size, grid_size, output_channels)
        x = x.permute(0, 2, 3, 1)

        return x


def train_detector(dataset_path, output_dir='models/detector', epochs=50,
                   batch_size=32, lr=1e-3, device='cuda'):
    """Train sprite detector."""

    print("="*80)
    print("SPRITE DETECTOR TRAINING")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)
    frames = data['frames']
    sprite_data = data['sprite_data']

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")
    print(f"Sample sprite count: {len(sprite_data[0])}")

    # Split train/val (90/10)
    n_train = int(0.9 * len(frames))
    train_frames = frames[:n_train]
    train_sprites = sprite_data[:n_train]
    val_frames = frames[n_train:]
    val_sprites = sprite_data[n_train:]

    print(f"\nTrain set: {len(train_frames)} frames")
    print(f"Val set: {len(val_frames)} frames")

    # Create datasets
    train_dataset = SpriteDataset(train_frames, train_sprites)
    val_dataset = SpriteDataset(val_frames, val_sprites)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    model = SpriteDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_has_loss = 0
        train_type_loss = 0
        train_offset_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for frames_batch, targets_batch in pbar:
            frames_batch = frames_batch.to(device)
            targets_batch = targets_batch.to(device)

            # Forward
            preds = model(frames_batch)

            # Loss calculation
            # 1. Binary cross-entropy for "has_sprite"
            has_sprite_pred = preds[:, :, :, 0]
            has_sprite_target = targets_batch[:, :, :, 0]
            has_loss = F.binary_cross_entropy_with_logits(has_sprite_pred, has_sprite_target)

            # 2. Cross-entropy for sprite type (only where sprite exists)
            type_pred = preds[:, :, :, 1:1+model.num_types]
            type_target = targets_batch[:, :, :, 1:1+model.num_types]
            mask = (has_sprite_target > 0.5).unsqueeze(-1)  # Only compute where sprite exists
            type_loss = F.binary_cross_entropy_with_logits(type_pred[mask.expand_as(type_pred)],
                                                            type_target[mask.expand_as(type_target)])

            # 3. MSE for position offsets (only where sprite exists)
            offset_pred = preds[:, :, :, 1+model.num_types:]
            offset_target = targets_batch[:, :, :, 1+model.num_types:]
            offset_loss = F.mse_loss(offset_pred[mask.expand_as(offset_pred)],
                                      offset_target[mask.expand_as(offset_target)])

            # Total loss
            loss = has_loss + type_loss + offset_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_has_loss += has_loss.item()
            train_type_loss += type_loss.item()
            train_offset_loss += offset_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'has': f'{has_loss.item():.4f}',
                'type': f'{type_loss.item():.4f}',
                'offset': f'{offset_loss.item():.4f}'
            })

        train_loss /= len(train_loader)
        train_has_loss /= len(train_loader)
        train_type_loss /= len(train_loader)
        train_offset_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_has_loss = 0
        val_type_loss = 0
        val_offset_loss = 0

        with torch.no_grad():
            for frames_batch, targets_batch in val_loader:
                frames_batch = frames_batch.to(device)
                targets_batch = targets_batch.to(device)

                preds = model(frames_batch)

                has_sprite_pred = preds[:, :, :, 0]
                has_sprite_target = targets_batch[:, :, :, 0]
                has_loss = F.binary_cross_entropy_with_logits(has_sprite_pred, has_sprite_target)

                type_pred = preds[:, :, :, 1:1+model.num_types]
                type_target = targets_batch[:, :, :, 1:1+model.num_types]
                mask = (has_sprite_target > 0.5).unsqueeze(-1)
                type_loss = F.binary_cross_entropy_with_logits(type_pred[mask.expand_as(type_pred)],
                                                                type_target[mask.expand_as(type_target)])

                offset_pred = preds[:, :, :, 1+model.num_types:]
                offset_target = targets_batch[:, :, :, 1+model.num_types:]
                offset_loss = F.mse_loss(offset_pred[mask.expand_as(offset_pred)],
                                          offset_target[mask.expand_as(offset_target)])

                loss = has_loss + type_loss + offset_loss

                val_loss += loss.item()
                val_has_loss += has_loss.item()
                val_type_loss += type_loss.item()
                val_offset_loss += offset_loss.item()

        val_loss /= len(val_loader)
        val_has_loss /= len(val_loader)
        val_type_loss /= len(val_loader)
        val_offset_loss /= len(val_loader)

        scheduler.step()

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Has: {train_has_loss:.4f}, Type: {train_type_loss:.4f}, Offset: {train_offset_loss:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Has: {val_has_loss:.4f}, Type: {val_type_loss:.4f}, Offset: {val_offset_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{output_dir}/best_detector.pth")
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{output_dir}/detector_epoch_{epoch+1}.pth")

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}/best_detector.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sprite detector")
    parser.add_argument("--dataset", type=str, default="detector_dataset.npz",
                       help="Path to dataset")
    parser.add_argument("--output", type=str, default="models/detector",
                       help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    train_detector(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
