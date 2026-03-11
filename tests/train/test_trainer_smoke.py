"""Trainer smoke test."""

from pathlib import Path

from graft_net.train.trainer import Trainer


def test_trainer_runs_one_train_and_eval_step(tmp_path: Path) -> None:
    trainer = Trainer.for_smoke_test(output_dir=tmp_path)
    metrics = trainer.run_smoke_epoch()
    assert "train/task_loss" in metrics
    assert "val/task_loss" in metrics
    assert metrics["train/task_loss"] >= 0
    assert metrics["val/task_loss"] >= 0


def test_trainer_checkpoint_roundtrip(tmp_path: Path) -> None:
    trainer = Trainer.for_smoke_test(output_dir=tmp_path)
    trainer.run_smoke_epoch()
    ckpt_path = trainer.save_checkpoint()
    assert ckpt_path.exists()
    # Reload
    trainer.load_checkpoint(ckpt_path)
