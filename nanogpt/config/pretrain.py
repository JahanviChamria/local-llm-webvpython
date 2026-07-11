# Stage 1: char-level pretrain on the full personal corpus + public VPython demos.
# ~10.7M params, sized for a 2.3M-char corpus on a Colab T4 (~15 min).

out_dir = 'out'
dataset = 'pretrain'

n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2

batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

eval_interval = 250
eval_iters = 100
log_interval = 50
always_save_checkpoint = False  # keep the best-val checkpoint, corpus is small
