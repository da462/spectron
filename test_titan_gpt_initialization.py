import unittest

import torch

from low_rank_linear import replace_linear_with_lowrank
from titan_gpt import TitanGPT, TitanModelArgs


class TestTitanGPTInitialization(unittest.TestCase):
    @staticmethod
    def _model_args(
        *,
        depth_init: bool,
        embedding_init_std: float | None = None,
    ) -> TitanModelArgs:
        return TitanModelArgs(
            vocab_size=4096,
            n_layers=2,
            n_heads=2,
            dim=128,
            max_seq_len=64,
            n_kv_heads=1,
            multiple_of=64,
            depth_init=depth_init,
            embedding_init_std=embedding_init_std,
        )

    def test_default_embedding_initialization_has_unit_std(self) -> None:
        torch.manual_seed(20260831)
        model = TitanGPT(self._model_args(depth_init=False))

        self.assertAlmostEqual(
            model.tok_embeddings.weight.std().item(), 1.0, places=2
        )

    def test_tt_style_embedding_initialization_has_unit_std(self) -> None:
        torch.manual_seed(20260831)
        model = TitanGPT(self._model_args(depth_init=True))

        self.assertAlmostEqual(
            model.tok_embeddings.weight.std().item(), 1.0, places=2
        )

    def test_embedding_std_override_changes_only_embeddings(self) -> None:
        torch.manual_seed(20260831)
        baseline = TitanGPT(self._model_args(depth_init=False))
        baseline_rng_state = torch.random.get_rng_state()
        torch.manual_seed(20260831)
        overridden = TitanGPT(
            self._model_args(depth_init=False, embedding_init_std=0.02)
        )
        overridden_rng_state = torch.random.get_rng_state()

        self.assertAlmostEqual(
            overridden.tok_embeddings.weight.std().item(), 0.02, places=3
        )
        torch.testing.assert_close(
            overridden.tok_embeddings.weight,
            baseline.tok_embeddings.weight * 0.02,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(baseline_rng_state, overridden_rng_state)
        overridden_params = dict(overridden.named_parameters())
        for name, parameter in baseline.named_parameters():
            if name == "tok_embeddings.weight":
                continue
            torch.testing.assert_close(
                parameter,
                overridden_params[name],
                rtol=0,
                atol=0,
            )

    def test_low_rank_conversion_does_not_replace_or_modify_embeddings(self) -> None:
        torch.manual_seed(20260831)
        model = TitanGPT(self._model_args(depth_init=True))
        embedding_parameter = model.tok_embeddings.weight
        embedding_before = embedding_parameter.detach().clone()

        converted = replace_linear_with_lowrank(
            model,
            rank_ratio=0.25,
            method="svd",
            exclude_modules=["tok_embeddings", "output"],
            disable_c=True,
        )

        self.assertIs(converted.tok_embeddings.weight, embedding_parameter)
        torch.testing.assert_close(
            converted.tok_embeddings.weight,
            embedding_before,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
