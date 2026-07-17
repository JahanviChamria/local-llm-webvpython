# local-llm-webvpython

Two small language models trained on every line of code I have written, evaluated on generating runnable Web VPython physics simulations. No API wrappers: one model is trained from scratch, the other is fine-tuned and served entirely on my laptop.

## Method

Two arms, same evaluation.

| | Arm A | Arm B |
|---|---|---|
| Base | nanoGPT, from scratch | Qwen2.5-Coder-0.5B-Instruct |
| Training | 2-stage char-level: pretrain on full corpus, finetune on sim mix | 2-phase LoRA (r=16): LM adaptation on full corpus, SFT on description-to-sim pairs |
| Size | ~10M params, vocab 335 chars | 0.5B params, Q4_K_M GGUF |
| Serving | PyTorch CPU | llama.cpp / `ollama run jahanvi-coder` |

## Corpus

- 497 deduplicated source files, 2.2 MB: 50 GitHub repos, glowscript.org programs, PyCharm projects, coursework.
- 6 unique Web VPython sims of mine plus 52 official GlowScript demo programs (public augmentation, since 6 sims alone would only teach memorization).
- Caveats: coursework folders include a starter scaffold that cannot be cleanly separated; one sim (ProjectileMotion2Dim) relies on GlowScript's JS looseness (`running` read before assignment) and fails strict CPython execution.

## Evaluation

pass@1 and pass@5 over 6 prompts (2 held-out sims never trained on + 4 novel physics prompts), 10 samples per prompt, temperature 0.8. A sample passes if it parses (`ast.parse`) and executes under a headless VPython stub (`eval/vpython_stub.py`) without exception, bounded at 300 animation steps, 20 s timeout.

Arm A training (T4, 2026-07-12): stage-1 val loss 5.86 -> 1.395 (best at step 3500 of 5000, best-val checkpoint kept as overfitting set in). Stage-2 finetune val 0.035, which mostly measures memorization of the oversampled sim mix; pass@k below is the honest metric.

Arm B training ran on the same T4 (2026-07-13). The training logs were lost to a Colab disconnect and the model was recovered from Drive, so its loss curve is unrecorded.

Results:

| Arm | pass@1 | pass@5 | code-seeded run rate |
|---|---|---|---|
| nanoGPT from scratch | 0.00 | 0.00 | 0.18 |
| Qwen2.5-Coder-0.5B + LoRA | 0.18 | 0.53 | n/a |

The from-scratch model cannot follow NL instructions — expected for 10M params on 2.3M chars, and the reason the eval includes a completion-style metric for it. Seeded with a code opening it produces runnable VPython 18% of the time (11/60), and 10/10 for a seed matching a training sim, which is memorization at work. The fine-tuned arm does follow instructions: 11/60 samples run (pass@1 0.183), and with 5 attempts a runnable sim comes out about half the time (pass@5 0.532). Per-prompt spread is wide, 5/10 on the pendulum prompt but 0/10 on bounce and spring. The contrast is the point: instruction following comes from the pretrained base, not from my 2.2 MB corpus, which at 10M params from scratch only buys memorization. Raw counts in `eval/results.json`.

## Pipeline

```
python scripts/scrape_glowscript.py            # my glowscript programs
python scripts/scrape_glowscript.py GlowScriptDemos vpython-public
python scripts/collect_repos.py                # all my repos -> corpus/raw/general
python scripts/build_dataset.py                # char bins + splits -> data/
python scripts/make_prompts.py                 # eval prompts + LoRA pairs
# train on Colab: notebooks/train_nanogpt_colab.ipynb, notebooks/lora_qwen_colab.ipynb
python eval/passk.py --arm a --arm b           # pass@k table
python demo/app.py                             # http://localhost:8155
```

Corpus and data files stay out of git (personal code); the scripts regenerate them.

## Demo

`demo/app.py` serves a prompt box on localhost. Pick an arm, generate, copy, paste into the glowscript.org editor, and the sim runs in the browser.
