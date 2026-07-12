# Stage 2: continue training on the VPython sim mix (her sims oversampled
# + Python-syntax GlowScript demos). Resumes the stage-1 checkpoint in out/.

out_dir = 'out'
dataset = 'finetune'
init_from = 'resume'

block_size = 256  # must match the pretrained model's context length
batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 1e-4
max_iters = 6000        # resume continues from ~5000, so ~1000 finetune iters
lr_decay_iters = 6000
min_lr = 1e-5
warmup_iters = 0

eval_interval = 100
eval_iters = 100
log_interval = 50
always_save_checkpoint = True  # val distribution changed, save the finetuned model
