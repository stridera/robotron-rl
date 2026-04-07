"""
Collect expert demonstrations from Brain3GymAdapter for behavioral cloning.

Runs N gym environments in a round-robin loop (single process, each with its own
Brain3GymAdapter instance) and records (obs, action) pairs.

Speed note: headless pygame renders off-screen at ~60 FPS even with fps=0.
Use --jobs to spawn multiple OS-level processes and merge results.

Output: demos/brain3_demos.npz  with arrays:
    obs       : float32  (N, 986)
    actions   : int32    (N, 2)   — [move_dir, shoot_dir] each 0-7
    rewards   : float32  (N,)
    dones     : bool     (N,)

Usage:
    # Quick test (100k, ~30 min single-process):
    poetry run python collect_brain3_demos.py --target 100000

    # Full collection (1M, parallel):
    poetry run python collect_brain3_demos.py --target 1000000 --jobs 8
"""
import argparse
import os
import time
import tempfile
import multiprocessing as mp
import numpy as np
from collections import defaultdict

from robotron import RobotronEnv
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from position_wrapper import GroundTruthPositionWrapper, OBS_DIM
from brain3_gym_adapter import Brain3GymAdapter


def make_env(config_path: str, level: int, seed: int, headless: bool = True):
    env = RobotronEnv(
        level=level,
        lives=5,
        fps=0,
        config_path=config_path,
        always_move=True,
        headless=headless,
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = GroundTruthPositionWrapper(env)
    obs, info = env.reset(seed=seed)
    return env, obs, info


def _worker(worker_id: int, target: int, config: str, envs_per_worker: int,
             start_level: int, out_path: str, counter=None):
    """Run a single worker process: collect `target` transitions and save to out_path."""
    import warnings
    warnings.filterwarnings('ignore')

    envs, obs_s, info_s, adapters = [], [], [], []
    num_curriculum_levels = 15
    levels = [(start_level + i) % num_curriculum_levels + 1 for i in range(envs_per_worker)]

    for i in range(envs_per_worker):
        env, obs, info = make_env(config, levels[i], seed=worker_id * 10000 + i * 1000)
        envs.append(env)
        obs_s.append(obs)
        info_s.append(info)
        adapters.append(Brain3GymAdapter())

    obs_buf    = np.zeros((target, OBS_DIM), dtype=np.float32)
    action_buf = np.zeros((target, 2),       dtype=np.int32)
    reward_buf = np.zeros(target,            dtype=np.float32)
    done_buf   = np.zeros(target,            dtype=bool)
    ep_scores  = []
    n = 0

    while n < target:
        for i in range(envs_per_worker):
            if n >= target:
                break
            obs  = obs_s[i]
            info = info_s[i]
            move_dir, shoot_dir = adapters[i].act(info)
            action = [move_dir, shoot_dir]

            obs_buf[n]    = obs
            action_buf[n] = action

            next_obs, reward, terminated, truncated, next_info = envs[i].step(action)
            done = terminated or truncated
            reward_buf[n] = reward
            done_buf[n]   = done
            n += 1
            if counter is not None:
                counter.value = n

            if done:
                ep_scores.append(next_info.get('score', 0))
                levels[i] = levels[i] % num_curriculum_levels + 1
                adapters[i].reset()
                next_obs, next_info = envs[i].reset()

            obs_s[i]  = next_obs
            info_s[i] = next_info

    for env in envs:
        env.close()

    np.savez_compressed(
        out_path,
        obs=obs_buf[:n],
        actions=action_buf[:n],
        rewards=reward_buf[:n],
        dones=done_buf[:n],
        episode_scores=np.array(ep_scores, dtype=np.float32),
    )
    avg_score = np.mean(ep_scores) if ep_scores else 0
    print(f"  [worker {worker_id}] Done: {n:,} transitions, {len(ep_scores)} episodes, "
          f"avg_score={avg_score:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',  type=int, default=1_000_000, help='Total transitions to collect')
    parser.add_argument('--envs',    type=int, default=4,          help='Envs per worker process')
    parser.add_argument('--jobs',    type=int, default=1,          help='Parallel worker processes')
    parser.add_argument('--config',  type=str, default='config.yaml')
    parser.add_argument('--output',  type=str, default='demos/brain3_demos.npz')
    parser.add_argument('--chunk',   type=int, default=100_000,    help='Save checkpoint every N transitions (single-job only)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("=" * 70)
    print("Brain3 Demo Collection")
    print("=" * 70)
    print(f"  Target transitions : {args.target:,}")
    print(f"  Envs per worker    : {args.envs}")
    print(f"  Worker processes   : {args.jobs}")
    print(f"  Config             : {args.config}")
    print(f"  Output             : {args.output}")
    print(f"  Obs dim            : {OBS_DIM}")
    print()

    t_start = time.time()

    if args.jobs == 1:
        # ── Single-process collection ─────────────────────────────────────────
        num_curriculum_levels = 15
        levels = [(i % num_curriculum_levels) + 1 for i in range(args.envs)]
        envs, obs_s, info_s, adapters = [], [], [], []
        for i in range(args.envs):
            env, obs, info = make_env(args.config, levels[i], seed=i * 1000)
            envs.append(env); obs_s.append(obs); info_s.append(info)
            adapters.append(Brain3GymAdapter())

        obs_buf     = np.zeros((args.target, OBS_DIM), dtype=np.float32)
        action_buf  = np.zeros((args.target, 2),        dtype=np.int32)
        reward_buf  = np.zeros(args.target,              dtype=np.float32)
        done_buf    = np.zeros(args.target,              dtype=bool)
        episode_scores, episode_levels = [], []
        action_counts = defaultdict(int)
        n_collected = n_episodes = 0
        t_last_print = last_checkpoint = 0

        while n_collected < args.target:
            for i in range(args.envs):
                if n_collected >= args.target:
                    break
                obs, info = obs_s[i], info_s[i]
                move_dir, shoot_dir = adapters[i].act(info)
                action = [move_dir, shoot_dir]
                obs_buf[n_collected]    = obs
                action_buf[n_collected] = action
                action_counts[(move_dir, shoot_dir)] += 1
                next_obs, reward, terminated, truncated, next_info = envs[i].step(action)
                done = terminated or truncated
                reward_buf[n_collected] = reward
                done_buf[n_collected]   = done
                n_collected += 1
                if done:
                    episode_scores.append(next_info.get('score', 0))
                    episode_levels.append(next_info.get('level', 0))
                    n_episodes += 1
                    levels[i] = levels[i] % num_curriculum_levels + 1
                    adapters[i].reset()
                    next_obs, next_info = envs[i].reset()
                obs_s[i], info_s[i] = next_obs, next_info

            t_now = time.time()
            elapsed = t_now - t_start
            if elapsed - t_last_print >= 10.0 or n_collected >= args.target:
                rate = n_collected / elapsed if elapsed > 0 else 0
                eta  = (args.target - n_collected) / rate if rate > 0 else 0
                avg_score = np.mean(episode_scores[-50:]) if episode_scores else 0
                print(f"  [{n_collected:>9,}/{args.target:,}]  "
                      f"rate={rate:,.0f}/s  ETA={eta/60:.1f}m  "
                      f"eps={n_episodes}  avg_score={avg_score:.0f}")
                t_last_print = elapsed
            if n_collected - last_checkpoint >= args.chunk and n_collected > 0:
                _save(args.output, obs_buf, action_buf, reward_buf, done_buf, n_collected, episode_scores)
                last_checkpoint = n_collected

        for env in envs:
            env.close()
        _save(args.output, obs_buf, action_buf, reward_buf, done_buf, n_collected, episode_scores)

    else:
        # ── Multi-process collection: each job writes its own tmp file ─────────
        per_job = args.target // args.jobs
        tmp_dir = tempfile.mkdtemp(prefix='brain3_demos_')
        print(f"  Collecting {per_job:,} transitions per worker (tmp: {tmp_dir})")
        print()

        procs = []
        out_paths = []
        counters = []
        for j in range(args.jobs):
            out_j = os.path.join(tmp_dir, f'worker_{j}.npz')
            out_paths.append(out_j)
            counter = mp.Value('i', 0)
            counters.append(counter)
            p = mp.Process(
                target=_worker,
                args=(j, per_job, args.config, args.envs,
                      (j * 3) % 15,  # stagger start levels
                      out_j, counter),
                daemon=True,
            )
            p.start()
            procs.append(p)
            print(f"  Spawned worker {j} (PID {p.pid})")

        print()
        # Monitor progress
        while any(p.is_alive() for p in procs):
            elapsed = time.time() - t_start
            alive = sum(p.is_alive() for p in procs)
            done_counts = [c.value for c in counters]
            total_done = sum(done_counts)
            total_target = per_job * args.jobs
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (total_target - total_done) / rate if rate > 0 else 0
            pct = 100.0 * total_done / total_target if total_target > 0 else 0
            print(f"  [{elapsed/60:.1f}m]  {total_done:,}/{total_target:,} ({pct:.1f}%)  "
                  f"rate={rate:,.0f}/s  ETA={eta/60:.1f}m  "
                  f"workers={alive}/{args.jobs}")
            time.sleep(30)

        for p in procs:
            p.join()

        # Merge all worker files
        print("\nMerging worker outputs...")
        all_obs, all_actions, all_rewards, all_dones, all_scores = [], [], [], [], []
        for out_j in out_paths:
            if not os.path.exists(out_j):
                print(f"  WARNING: {out_j} missing!")
                continue
            d = np.load(out_j)
            all_obs.append(d['obs'])
            all_actions.append(d['actions'])
            all_rewards.append(d['rewards'])
            all_dones.append(d['dones'])
            all_scores.extend(d['episode_scores'].tolist())

        merged_obs     = np.concatenate(all_obs)
        merged_actions = np.concatenate(all_actions)
        merged_rewards = np.concatenate(all_rewards)
        merged_dones   = np.concatenate(all_dones)
        np.savez_compressed(
            args.output,
            obs=merged_obs, actions=merged_actions,
            rewards=merged_rewards, dones=merged_dones,
            episode_scores=np.array(all_scores, dtype=np.float32),
        )
        n_collected = len(merged_obs)
        episode_scores = all_scores
        action_counts = defaultdict(int)
        for a in merged_actions:
            action_counts[(int(a[0]), int(a[1]))] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"Done!  {n_collected:,} transitions in {elapsed/60:.1f} min  "
          f"({n_collected/elapsed:,.0f}/sec)")
    if episode_scores:
        print(f"Episodes: {len(episode_scores)}  "
              f"Avg score: {np.mean(episode_scores):.0f}  "
              f"Max score: {max(episode_scores)}")
    print(f"Saved: {args.output}")

    print("\nTop 10 action pairs (move, shoot):")
    for (m, s), cnt in sorted(action_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  move={m} shoot={s}: {cnt:,} ({100.0*cnt/n_collected:.1f}%)")


def _save(path, obs_buf, action_buf, reward_buf, done_buf, n, episode_scores):
    np.savez_compressed(
        path,
        obs=obs_buf[:n],
        actions=action_buf[:n],
        rewards=reward_buf[:n],
        dones=done_buf[:n],
        episode_scores=np.array(episode_scores, dtype=np.float32),
    )


n_collected = 0  # module-level for summary block when using multiprocessing path


if __name__ == '__main__':
    main()
