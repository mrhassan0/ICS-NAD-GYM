"""Phase 3: RL-only baseline. Trains RecurrentPPO (sb3-contrib, MlpLstmPolicy) on
ICSNADGym with the base reward table (no LLM shaping). This is arm (b) of the eventual
3-way comparison against the paper's supervised baselines and the future RL+LLM arm.

Resumable by design: this environment's session can be torn down mid-run without warning
(observed repeatedly — background training processes vanish cleanly, no traceback, after
roughly 10-15 minutes of wall clock time). Every invocation checks for the latest checkpoint
under the run's checkpoints/ dir and continues from it rather than starting over, so
re-running the exact same command after an unexpected teardown just picks up where it left off.
"""

import argparse
import glob
import os
import re
import sys

try:
    import fcntl
except ImportError:
    fcntl = None

# Must be set before torch is imported anywhere (including transitively via sb3_contrib) —
# by default PyTorch's CPU intra-op thread pool tries to use every core on the machine.
# Running several training processes concurrently (multi-seed replication) means each one
# independently grabs all 32 cores for its own backprop step, causing catastrophic
# oversubscription during the optimizer phase specifically — observed as training stalling
# at exactly the same step count (right after the first rollout/update cycle) across every
# concurrent run, reproducibly, even with no other jobs on the machine. Capping each
# process's thread count keeps total demand reasonable when running several at once.
_torch_threads = "4"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--torch-threads" and _i + 1 < len(sys.argv):
        _torch_threads = sys.argv[_i + 1]
os.environ.setdefault("OMP_NUM_THREADS", _torch_threads)
os.environ.setdefault("MKL_NUM_THREADS", _torch_threads)

from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from icsnad_gym.env import ICSNADGym

EXPERIMENTS_DIR = Path("/home/user5/Desktop/MS THESIS/experiments")
DEFAULT_N_ENVS = 8
DEFAULT_MAX_CONCURRENT_RUNS = 2
DEFAULT_MAX_TOTAL_ENVS = 16
_RUN_LOCK_HANDLES = []


def _is_recurrent_ppo_command(argv: list[str]) -> bool:
    if not argv or "python" not in Path(argv[0]).name:
        return False
    for i, arg in enumerate(argv):
        if arg == "-m" and i + 1 < len(argv) and argv[i + 1] == "rl.train_recurrent_ppo":
            return True
        if arg.endswith("train_recurrent_ppo.py"):
            return True
    return False


def _int_arg(argv: list[str], name: str, default: int) -> int:
    prefix = f"{name}="
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
        if arg.startswith(prefix):
            try:
                return int(arg[len(prefix):])
            except ValueError:
                return default
    return default


def _active_training_runs() -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return runs

    current_pid = os.getpid()
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw_cmdline:
            continue

        argv = [part.decode(errors="replace") for part in raw_cmdline.split(b"\0") if part]
        if _is_recurrent_ppo_command(argv):
            n_envs = _int_arg(argv, "--n-envs", DEFAULT_N_ENVS)
            runs.append((pid, n_envs, " ".join(argv[:8])))
    return runs


def _enforce_concurrency_guard(args: argparse.Namespace) -> None:
    if args.skip_concurrency_guard:
        return

    active_runs = _active_training_runs()
    planned_run_count = len(active_runs) + 1
    planned_env_count = sum(n_envs for _, n_envs, _ in active_runs) + args.n_envs

    exceeds_run_limit = (
        args.max_concurrent_runs > 0
        and planned_run_count > args.max_concurrent_runs
    )
    exceeds_env_limit = (
        args.max_total_envs > 0
        and planned_env_count > args.max_total_envs
    )
    if not (exceeds_run_limit or exceeds_env_limit):
        _acquire_run_slot(args.max_concurrent_runs)
        return

    active_summary = "\n".join(
        f"  pid={pid}, n_envs={n_envs}, cmd={cmd}"
        for pid, n_envs, cmd in active_runs
    ) or "  none"
    raise SystemExit(
        "Refusing to start another RecurrentPPO run because it would exceed the "
        "local memory-safety guard.\n"
        f"Planned runs: {planned_run_count} "
        f"(limit {args.max_concurrent_runs}; 0 disables)\n"
        f"Planned vector envs: {planned_env_count} "
        f"(limit {args.max_total_envs}; 0 disables)\n"
        "Active runs:\n"
        f"{active_summary}\n"
        "Run fewer seeds at once, lower --n-envs, or pass "
        "--skip-concurrency-guard only if you have checked available RAM."
    )


def _acquire_run_slot(max_concurrent_runs: int) -> None:
    if max_concurrent_runs <= 0 or fcntl is None:
        return

    lock_dir = EXPERIMENTS_DIR / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(max_concurrent_runs):
        lock_file = lock_dir / f"train_recurrent_ppo_{slot}.lock"
        handle = lock_file.open("w")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue

        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        _RUN_LOCK_HANDLES.append(handle)
        return

    raise SystemExit(
        "Refusing to start another RecurrentPPO run because all local run slots "
        f"are busy (limit {max_concurrent_runs}). Wait for a run to finish, or "
        "pass --skip-concurrency-guard only if you have checked available RAM."
    )


def make_env(split: str, episode_length: int, seed: int, use_llm_shaping: bool = False):
    def _init():
        return ICSNADGym(split=split, episode_length=episode_length, seed=seed, use_llm_shaping=use_llm_shaping)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--run-name", type=str, default="recurrentppo_baseline_v1")
    parser.add_argument("--checkpoint-freq", type=int, default=4_096,
                         help="small default: session teardowns have hit every ~10-15 min of "
                              "wall clock, so checkpoints need to land well inside that window")
    parser.add_argument("--torch-threads", type=int, default=4,
                         help="caps PyTorch CPU intra-op threads; parsed pre-import at module level too")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--net-arch", type=int, default=64, help="hidden units per MLP layer")
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--use-llm-shaping", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                         help="controls both RecurrentPPO's weight init/training stochasticity "
                              "and offsets per-env episode-sampling seeds, for multi-seed replication")
    parser.add_argument("--max-concurrent-runs", type=int, default=DEFAULT_MAX_CONCURRENT_RUNS,
                        help="memory-safety guard; 0 disables the run-count limit")
    parser.add_argument("--max-total-envs", type=int, default=DEFAULT_MAX_TOTAL_ENVS,
                        help="memory-safety guard across active train_recurrent_ppo runs; 0 disables")
    parser.add_argument("--skip-concurrency-guard", action="store_true",
                        help="override the local memory-safety guard")
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)

    run_dir = EXPERIMENTS_DIR / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    tensorboard_dir = EXPERIMENTS_DIR / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    _enforce_concurrency_guard(args)

    # start_method="spawn" (not the Linux default "fork"): polars/pyarrow maintain internal
    # Rust thread pools that are unsafe to inherit across fork() — forking a process that
    # already used polars causes intermittent native segfaults (observed: crashed at ~40%
    # through a training run with no Python-level traceback, exit code 139/SIGSEGV).
    vec_env = SubprocVecEnv(
        [make_env("train", args.episode_length, seed=args.seed * 100 + i, use_llm_shaping=args.use_llm_shaping)
         for i in range(args.n_envs)],
        start_method="spawn",
    )

    # Resume from the latest checkpoint if one exists (see module docstring — session
    # teardowns have killed background training runs mid-flight with no error, repeatedly).
    # Also consider a prior final_model.zip: a run that finished its target and saved via
    # model.save() can be well past its last numbered checkpoint (checkpoint_freq doesn't
    # necessarily align with total_timesteps) — extending that run's target later must resume
    # from the true final state, not silently rewind and redo already-completed steps.
    checkpoint_files = glob.glob(str(run_dir / "checkpoints" / "recurrentppo_*_steps.zip"))
    latest_checkpoint, latest_steps = None, 0
    for f in checkpoint_files:
        m = re.search(r"_(\d+)_steps\.zip$", f)
        if m and int(m.group(1)) > latest_steps:
            latest_checkpoint, latest_steps = f, int(m.group(1))

    final_model_path = run_dir / "final_model.zip"
    if final_model_path.exists():
        final_steps = RecurrentPPO.load(str(final_model_path), device="cpu").num_timesteps
        if final_steps > latest_steps:
            latest_checkpoint, latest_steps = str(final_model_path), final_steps

    # device="cpu": diagnosed a reproducible CUDA/cuDNN-LSTM segfault on this PyTorch
    # 2.12/CUDA 13.0 stack (crashed at 3 different, non-deterministic step counts across
    # fork/spawn/single-process configs — isolated to CUDA specifically via a clean 200k-step
    # CPU-only run). This policy network is small (MLP+LSTM) and doesn't need GPU
    # acceleration; the GPU is reserved for the LLM component in later phases anyway.
    if latest_checkpoint:
        print(f"Resuming from {latest_checkpoint} ({latest_steps:,} steps already done)")
        model = RecurrentPPO.load(latest_checkpoint, env=vec_env, device="cpu")
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            n_steps=args.n_steps,
            batch_size=args.n_steps * args.n_envs // 8 if args.n_envs >= 8 else args.n_steps,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=args.ent_coef,
            device="cpu",
            seed=args.seed,
            policy_kwargs=dict(
                net_arch=dict(pi=[args.net_arch, args.net_arch], vf=[args.net_arch, args.net_arch]),
                lstm_hidden_size=args.lstm_hidden_size,
            ),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="recurrentppo",
    )

    remaining = max(args.total_timesteps - latest_steps, 0)
    if remaining == 0:
        print(f"Already at or past target ({latest_steps:,} >= {args.total_timesteps:,}), skipping training.")
    else:
        model.learn(
            total_timesteps=remaining,
            callback=checkpoint_cb,
            tb_log_name=args.run_name,
            reset_num_timesteps=(latest_checkpoint is None),
        )
    model.save(str(run_dir / "final_model"))
    print(f"Training complete. Final model saved to {run_dir / 'final_model'}")


if __name__ == "__main__":
    main()
