"""Step 10: Offline RL fine-tuning from recorded Caldera emulation experiences.

Loads one or more .npz experience files (produced by CalderaExperienceRecorder.save),
adds the transitions into the agent's replay buffer, runs a fixed number of gradient
steps, and saves the updated checkpoint.

The agent is loaded from an existing checkpoint, fine-tuned in-process, then saved
to a new checkpoint path so the original is preserved.

Typical CLI call (from run_menu.py menu option h):
    python -m red.caldera.retrain \
        --checkpoint checkpoints/general_v3.pt \
        --experiences runs/caldera/*/experience.npz \
        --out checkpoints/general_v3_ft1.pt \
        --steps 500

Training note:
  Emulation episodes are short and the reward is approximated, so many small updates
  beat a few large ones. The default 500 gradient steps matches one CAGE-4 episode.
  Call repeatedly after each Caldera run to accumulate knowledge over time.
"""
import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from red.caldera.experience import CalderaExperienceRecorder
from red.agents.rl import MultiHeadDDQNAgent


def retrain(
    agent: MultiHeadDDQNAgent,
    experience_paths: List[Union[str, Path]],
    n_steps: int = 500,
    log=print,
) -> dict:
    """Add emulation experiences to the agent's replay buffer and run gradient steps.

    Returns a dict with the number of transitions loaded and the final loss.

    The caller is responsible for saving the agent afterwards.
    """
    total_added = 0
    for path in experience_paths:
        transitions = CalderaExperienceRecorder.load(Path(path))
        if not transitions:
            log(f'  {path}: kosong, dilewati.')
            continue
        for t in transitions:
            agent.replay.add(t.feature, t.reward, t.next_features, t.done, t.duration, t.mask)
        total_added += len(transitions)
        log(f'  {path}: {len(transitions)} transisi dimuat.')

    if total_added == 0:
        log('Tidak ada transisi yang dimuat. Pelatihan dilewati.')
        return {'transitions_loaded': 0, 'steps_run': 0, 'final_loss': None}

    need = max(agent.cfg.batch_size, agent.cfg.warmup)
    if len(agent.replay) < need:
        log(f'Transisi di buffer {len(agent.replay)} < {need} (batch/warmup). Tidak ada langkah gradien.')
        return {'transitions_loaded': total_added, 'steps_run': 0, 'final_loss': None}
    log(f'Total {total_added} transisi di buffer. Menjalankan {n_steps} langkah gradien…')
    last_loss = None
    steps_run = 0
    for _ in range(n_steps):
        loss = agent.train_step()
        if loss is not None:
            last_loss = loss
            steps_run += 1

    log(f'Selesai: {steps_run} langkah, loss terakhir = {last_loss}')
    return {'transitions_loaded': total_added, 'steps_run': steps_run, 'final_loss': last_loss}


def retrain_from_checkpoint(
    checkpoint_path: Union[str, Path],
    experience_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    n_steps: int = 500,
    log=print,
) -> Optional[Path]:
    """Load a checkpoint, fine-tune on emulation experiences, save to output_path.

    Raises FileExistsError if output_path already exists (safe by default).
    Returns the path where the new checkpoint was saved, or None when no gradient step ran (nothing is saved).
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'{output_path} sudah ada; pilih nama lain.')

    log(f'Memuat checkpoint dari {checkpoint_path.name}…')
    agent = MultiHeadDDQNAgent.load(checkpoint_path)
    agent.online.train()

    stats = retrain(agent, experience_paths, n_steps=n_steps, log=log)
    if stats['steps_run'] == 0:
        log('Tidak ada langkah gradien: checkpoint tidak disimpan.')
        return None

    extra = {
        'fine_tuned_from': str(checkpoint_path),
        'experiences': [str(p) for p in experience_paths],
        'transitions_loaded': stats['transitions_loaded'],
        'steps_run': stats['steps_run'],
        'final_loss': stats['final_loss'],
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    agent.save(output_path, extra=extra)
    log(f'Checkpoint baru disimpan ke {output_path}')
    return output_path


def retrain_after_run(runs_root, checkpoint_path, output_path, log=print) -> Optional[Path]:
    """Step 10, automatic (figure 6): after a RL+LLM run that accepted LLM actions, fine-tune on the experiences of all
    Caldera runs under runs_root. The result is a candidate file at output_path; the checkpoint in use is not changed.
    Returns None when there are no experiences or no gradient step could run."""
    paths = sorted(Path(runs_root).glob('*/experience.npz'))
    if not paths:
        return None
    return retrain_from_checkpoint(checkpoint_path, paths, output_path, log=log)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description='Fine-tune RL agent from Caldera emulation experiences.')
    p.add_argument('--checkpoint', required=True, help='Input checkpoint (.pt)')
    p.add_argument('--experiences', nargs='+', required=True,
                   help='Experience files or glob patterns (.npz)')
    p.add_argument('--out', required=True, help='Output checkpoint path (.pt)')
    p.add_argument('--steps', type=int, default=500, help='Gradient steps (default 500)')
    args = p.parse_args()

    paths = []
    for pattern in args.experiences:
        matched = glob.glob(pattern, recursive=True)
        paths.extend(matched if matched else [pattern])

    retrain_from_checkpoint(args.checkpoint, paths, args.out, n_steps=args.steps)


if __name__ == '__main__':
    _main()
