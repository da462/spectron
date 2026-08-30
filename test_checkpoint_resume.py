import argparse
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simple_gpt_dataloader import OffsetSequentialSampler, train_batch_offset
from simple_gpt_training import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_resume_compatibility,
)


def make_offset_loader(dataset, batch_size, batches_consumed):
    _, sample_offset = train_batch_offset(
        batches_consumed,
        len(dataset),
        batch_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=OffsetSequentialSampler(dataset, sample_offset),
    )


def take_cyclic_batches(dataset, batch_size, batches_consumed, count):
    loader = make_offset_loader(dataset, batch_size, batches_consumed)
    iterator = iter(loader)
    batches = []
    for _ in range(count):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(make_offset_loader(dataset, batch_size, 0))
            batch = next(iterator)
        batches.append(batch[0].clone())
    return batches


def make_train_state():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 6),
        torch.nn.GELU(),
        torch.nn.Linear(6, 3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.05 * step,
    )
    return model, optimizer, scheduler


def train_one_random_step(model, optimizer, scheduler):
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach().clone(), inputs.clone(), targets.clone()


class DataCursorTest(unittest.TestCase):
    def test_resumed_batches_match_uninterrupted_batches_across_wrap(self):
        dataset = TensorDataset(torch.arange(11))
        batch_size = 3
        consumed = 5

        uninterrupted = take_cyclic_batches(dataset, batch_size, 0, consumed + 9)
        resumed = take_cyclic_batches(dataset, batch_size, consumed, 9)

        self.assertEqual(len(resumed), 9)
        for expected, actual in zip(uninterrupted[consumed:], resumed):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_offset_accounts_for_partial_final_batch(self):
        self.assertEqual(train_batch_offset(0, dataset_size=11, batch_size=3), (0, 0))
        self.assertEqual(train_batch_offset(3, dataset_size=11, batch_size=3), (3, 9))
        self.assertEqual(train_batch_offset(4, dataset_size=11, batch_size=3), (0, 0))
        self.assertEqual(train_batch_offset(7, dataset_size=11, batch_size=3), (3, 9))


class CheckpointResumeTest(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    def test_rng_round_trip_covers_python_numpy_and_torch(self):
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(4))

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_checkpoint_resume_matches_uninterrupted_optimizer_step(self):
        args = argparse.Namespace(
            batch_size=8,
            micro_batch_size=2,
            train_seq_len=16,
            train_files="train.bin",
            virtual_workers_per_gpu=1,
            seed=123,
            optimizer="adamw",
            scheduler="cosine",
            lr_schedule_steps=20,
            warmup_steps=2,
            max_lr=0.03,
            weight_decay=0.1,
        )
        model, optimizer, scheduler = make_train_state()
        for _ in range(3):
            train_one_random_step(model, optimizer, scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            rng_state = capture_rng_state()
            checkpoint_path = save_checkpoint(
                checkpoint_dir=tmpdir,
                model=model,
                model_args=argparse.Namespace(name="tiny"),
                shared_optimizer=optimizer,
                private_optimizers={},
                private_param_store={},
                scheduler=scheduler,
                step=2,
                total_tokens_rank0=384,
                total_tokens_world=384,
                total_flops=12,
                total_flops_forward=4,
                total_flops_backward=8,
                args=args,
                train_batches_consumed_per_worker=12,
                accumulation_steps=4,
                world_size=1,
                virtual_world_size=1,
                rng_states_by_rank=[rng_state],
            )
            self.assertTrue(Path(checkpoint_path).is_file())
            self.assertEqual(list(Path(tmpdir).glob("*.tmp-*")), [])

            expected_loss, expected_inputs, expected_targets = train_one_random_step(
                model, optimizer, scheduler
            )
            expected_lr = scheduler.get_last_lr()[0]
            expected_parameters = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }

            # Recreating the objects intentionally consumes RNG before load.
            resumed_model, resumed_optimizer, resumed_scheduler = make_train_state()
            metadata = load_checkpoint(
                checkpoint_path,
                resumed_model,
                resumed_optimizer,
                {},
                {},
                resumed_scheduler,
                device="cpu",
                args=args,
            )
            restore_rng_state(metadata["rng_state"])
            actual_loss, actual_inputs, actual_targets = train_one_random_step(
                resumed_model, resumed_optimizer, resumed_scheduler
            )

            self.assertEqual(metadata["start_step"], 3)
            self.assertEqual(metadata["train_batches_consumed_per_worker"], 12)
            self.assertEqual(metadata["saved_accumulation_steps"], 4)
            torch.testing.assert_close(actual_inputs, expected_inputs, rtol=0, atol=0)
            torch.testing.assert_close(actual_targets, expected_targets, rtol=0, atol=0)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            self.assertEqual(resumed_scheduler.get_last_lr()[0], expected_lr)
            for name, parameter in resumed_model.named_parameters():
                torch.testing.assert_close(
                    parameter,
                    expected_parameters[name],
                    rtol=0,
                    atol=0,
                )

    def test_incompatible_batch_configuration_is_rejected(self):
        checkpoint = {
            "args": {"batch_size": 512, "micro_batch_size": 16},
            "resume_state": {"world_size": 4, "virtual_world_size": 4},
        }
        current_args = argparse.Namespace(batch_size=256, micro_batch_size=16)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_resume_compatibility(
                checkpoint,
                current_args,
                world_size=4,
                virtual_world_size=4,
            )

    def test_incompatible_world_size_is_rejected(self):
        checkpoint = {
            "args": {},
            "resume_state": {"world_size": 4, "virtual_world_size": 4},
        }
        with self.assertRaisesRegex(ValueError, "world_size"):
            validate_resume_compatibility(
                checkpoint,
                argparse.Namespace(),
                world_size=8,
                virtual_world_size=8,
            )


if __name__ == "__main__":
    unittest.main()
