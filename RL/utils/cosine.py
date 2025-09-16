import numpy as np

def cosine_lr_schedule(initial_lr, min_lr=1e-5, progress_remaining=1.0, warmup_proportion=0.03):
    """
    Computes the cosine decay of the learning rate with a linear warmup period at the beginning.

    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :param progress_remaining: The progress remaining (from 1 to 0).
    :param warmup_proportion: The proportion of the total training time to be used for linear warmup.
    :return: The adjusted learning rate based on the cosine schedule with warmup.
    """
    # Ensure progress_remaining is between 0 and 1
    progress_remaining = np.clip(progress_remaining, 0, 1)

    if progress_remaining > (1 - warmup_proportion):
        # Warmup phase: linearly increase LR
        warmup_progress = (1 - progress_remaining) / warmup_proportion
        new_lr = min_lr + (initial_lr - min_lr) * warmup_progress
    else:
        # Adjusted progress considering warmup phase
        adjusted_progress = (progress_remaining - (1 - warmup_proportion)) / (1 - warmup_proportion)

        # Cosine decay phase
        cos_decay = 0.5 * (1 + np.cos(np.pi * adjusted_progress))
        decayed = (1 - min_lr / initial_lr) * cos_decay + min_lr / initial_lr
        new_lr = initial_lr * decayed

    return new_lr