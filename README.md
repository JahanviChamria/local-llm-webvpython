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
- Caveats: coursework folders include professor-written starter scaffold that cannot be cleanly separated; one sim (ProjectileMotion2Dim) relies on GlowScript's JS looseness (`running` read before assignment) and fails strict CPython execution.

## Evaluation

pass@1 and pass@5 over 6 prompts (2 held-out sims never trained on + 4 novel physics prompts), 10 samples per prompt, temperature 0.8. A sample passes if it parses (`ast.parse`) and executes under a headless VPython stub (`eval/vpython_stub.py`) without exception, bounded at 300 animation steps, 20 s timeout.

Arm A training (T4, 2026-07-12): stage-1 val loss 5.86 -> 1.395 (best at step 3500 of 5000, best-val checkpoint kept as overfitting set in). Stage-2 finetune val 0.035, which mostly measures memorization of the oversampled sim mix; pass@k below is the honest metric.

Results: **in progress** — table lands here after the arm B run and eval.

| Arm | pass@1 | pass@5 |
|---|---|---|
| nanoGPT from scratch | – | – |
| Qwen2.5-Coder-0.5B + LoRA | – | – |

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
