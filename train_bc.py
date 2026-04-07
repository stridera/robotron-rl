"""
Behavioral Cloning from Brain3 expert demonstrations.

Trains an SB3 PPO policy using supervised learning (maximize log-likelihood
of expert actions). Produces a warm-started policy that mimics Brain3's
strategy before any RL fine-tuning.

Expected outcome:
  - After 30-50 epochs on 1M demos: W3-W5 performance without RL
  - Use models/bc_init/ as --bc-checkpoint in train_progressive.py

Usage:
    # Train BC model:
    poetry run python train_bc.py --demos demos/brain3_demos.npz

    # Quick test on small demo file:
    poetry run python train_bc.py --demos demos/test_demos.npz --epochs 5 --batch 64
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robotron import RobotronEnv
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from position_wrapper import GroundTruthPositionWrapper, OBS_DIM


def make_dummy_env(config_path: str = 'progressive_curriculum.yaml'):
    """Create a single env for PPO model initialization (never actually used for RL)."""
    def _init():
        env = RobotronEnv(level=1, lives=1, fps=0, always_move=True, headless=True,
                          config_path=config_path)
        env = MultiDiscreteToDiscrete(env)
        env = FrameSkipWrapper(env, skip=4)
        env = GroundTruthPositionWrapper(env)
        return env
    return _init


def build_ppo_model(env, device: str, net_arch: list = None):
    """Initialize a PPO model with the BC architecture."""
    if net_arch is None:
        net_arch = [512, 512]

    model = PPO(
        policy='MlpPolicy',
        env=env,
        device=device,
        policy_kwargs={'net_arch': net_arch},
        verbose=0,
    )
    return model


def evaluate_bc_policy(model, demos_path: str, device: str, n_samples: int = 10000):
    """Evaluate BC policy: return mean log-prob and action accuracy on held-out samples."""
    data = np.load(demos_path)
    n = min(n_samples, len(data['obs']))
    idx = np.random.choice(len(data['obs']), n, replace=False)
    obs_np     = data['obs'][idx].astype(np.float32)
    action_np  = data['actions'][idx].astype(np.int64)

    obs_t    = torch.as_tensor(obs_np, device=device)
    action_t = torch.as_tensor(action_np, device=device)

    model.policy.eval()
    with torch.no_grad():
        _, log_prob, _ = model.policy.evaluate_actions(obs_t, action_t)
        loss = -log_prob.mean().item()

        # Accuracy: compare argmax of predicted distribution to expert
        features = model.policy.extract_features(obs_t)
        latent_pi, _ = model.policy.mlp_extractor(features)
        dist = model.policy._get_action_dist_from_latent(latent_pi)
        # SB3 MultiCategoricalDistribution uses .distribution (list of Categoricals)
        pred_move  = dist.distribution[0].probs.argmax(dim=-1)
        pred_shoot = dist.distribution[1].probs.argmax(dim=-1)
        acc_move   = (pred_move  == action_t[:, 0]).float().mean().item()
        acc_shoot  = (pred_shoot == action_t[:, 1]).float().mean().item()

    model.policy.train()
    return loss, acc_move, acc_shoot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos',    type=str, default='demos/brain3_demos.npz')
    parser.add_argument('--output',   type=str, default='models/bc_init')
    parser.add_argument('--epochs',   type=int, default=40)
    parser.add_argument('--batch',    type=int, default=256)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--net-arch', type=str, default='512,512',
                        help='Comma-separated layer sizes, e.g. 512,512')
    parser.add_argument('--device',   type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--config',   type=str, default='progressive_curriculum.yaml')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    args = parser.parse_args()

    net_arch = [int(x) for x in args.net_arch.split(',')]
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("Behavioral Cloning from Brain3 Demos")
    print("=" * 70)
    print(f"  Demos    : {args.demos}")
    print(f"  Output   : {args.output}")
    print(f"  Device   : {args.device}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  LR       : {args.lr}")
    print(f"  Net arch : {net_arch}")
    print(f"  Obs dim  : {OBS_DIM}")
    print()

    # ── Load demos ────────────────────────────────────────────────────────────
    print("Loading demonstrations...", end=' ', flush=True)
    data = np.load(args.demos)
    obs_np    = data['obs'].astype(np.float32)
    action_np = data['actions'].astype(np.int64)
    n_demos   = len(obs_np)
    print(f"{n_demos:,} transitions loaded.")

    if 'episode_scores' in data and len(data['episode_scores']) > 0:
        scores = data['episode_scores']
        print(f"  Episode scores: mean={scores.mean():.0f}  max={scores.max():.0f}  "
              f"n_eps={len(scores)}")
    print()

    # ── Build model ───────────────────────────────────────────────────────────
    print("Building PPO model...", end=' ', flush=True)
    vec_env = DummyVecEnv([make_dummy_env(args.config)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
    model = build_ppo_model(vec_env, args.device, net_arch)
    print(f"Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    # ── DataLoader ────────────────────────────────────────────────────────────
    obs_t    = torch.as_tensor(obs_np)
    action_t = torch.as_tensor(action_np)
    dataset  = TensorDataset(obs_t, action_t)
    loader   = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2,
                          pin_memory=(args.device != 'cpu'))

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.policy.train()
    best_loss = float('inf')
    t_start = time.time()

    print(f"{'Epoch':>6}  {'Loss':>8}  {'Acc(mv)':>8}  {'Acc(sh)':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for obs_batch, action_batch in loader:
            obs_batch    = obs_batch.to(args.device)
            action_batch = action_batch.to(args.device)

            # SB3 evaluate_actions: obs → log_prob of given actions
            _, log_prob, entropy = model.policy.evaluate_actions(obs_batch, action_batch)
            bc_loss = -log_prob.mean()

            optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()

            epoch_loss += bc_loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t_start
        lr_now   = scheduler.get_last_lr()[0]

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            eval_loss, acc_mv, acc_sh = evaluate_bc_policy(
                model, args.demos, args.device, n_samples=min(20000, n_demos)
            )
            print(f"{epoch:>6}  {avg_loss:>8.4f}  {acc_mv:>7.1%}  {acc_sh:>7.1%}  "
                  f"{lr_now:>8.2e}  {elapsed/60:>5.1f}m")

            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.policy.save(os.path.join(args.output, 'bc_policy_best.pt'))
        else:
            print(f"{epoch:>6}  {avg_loss:>8.4f}  {'---':>8}  {'---':>8}  "
                  f"{lr_now:>8.2e}  {elapsed/60:>5.1f}m")

    # ── Save final model ──────────────────────────────────────────────────────
    # Save as full SB3-compatible PPO zip (load with PPO.load())
    model.save(os.path.join(args.output, 'bc_model'))
    model.policy.save(os.path.join(args.output, 'bc_policy_final.pt'))

    # Also save VecNormalize stats (identity — BC doesn't normalize obs)
    vec_env.save(os.path.join(args.output, 'vec_normalize.pkl'))

    vec_env.close()

    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"BC training complete in {elapsed/60:.1f} min")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output}/")
    print()
    print("Next step:")
    print(f"  poetry run python train_progressive.py --bc-checkpoint {args.output}/bc_model.zip")


if __name__ == '__main__':
    main()
