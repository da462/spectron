# Keller Muon Cooldown Protocol

This branch adds an opt-in training protocol for controlled Muon experiments:

- hidden 2D+ matrices use Keller/Jordan Muon LR adjustment (`original`);
- embedding, output, gain, and bias parameters use auxiliary AdamW at `0.1x`
  the scheduled Muon base LR;
- training starts at the peak LR without warmup;
- LR stays at its peak for the first 70% of optimizer updates;
- LR decays linearly to zero over the final 30%;
- heavy mechanistic and lightweight update diagnostics remain available.

The protocol is inspired by the flat-then-cooldown structure used in Modded
NanoGPT. It is intentionally named `stable_linear_decay`, because the current
Modded NanoGPT recipe also contains other schedule and batch-size choices that
this experiment does not reproduce.

Prepare the default low-rank FFN run without submitting:

```bash
DRY_RUN=1 bin/submit_jz_keller_cooldown.sh a100_4_dev2h_cpu30_whj lowrank_ffn
```

Submit by setting `DRY_RUN=0`. The defaults are peak Muon LR `5e-2`, auxiliary
AdamW LR `5e-3`, WD `0.01`, embedding init std `0.02`, and 2,234 updates. All
values can be overridden through the environment.

References:

- https://github.com/KellerJordan/Muon
- https://github.com/KellerJordan/modded-nanogpt
