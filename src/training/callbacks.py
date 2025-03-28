import time

import deepspeed
import torch.distributed as dist
from transformers import TrainerCallback


class PrinterCallback(TrainerCallback):
    time_before = time.time()
    step_before = 0

    def __init__(self, model, barrier=True):
        self.model = model
        self.barrier = barrier

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs["TFLOP/s"] = self.throughput_calculator(args, state)

    def throughput_calculator(self, args, state):
        if self.barrier:
            dist.barrier()
            deepspeed.get_accelerator().synchronize()

        time_current = time.time()
        step_current = state.global_step
        elapsed_time_per_step = (time_current - self.time_before) / (
            step_current - self.step_before
        )
        self.time_before = time_current
        self.step_before = step_current

        world_size = dist.get_world_size()
        batch_size = args.per_device_train_batch_size * world_size

        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        vocab_size = self.model.config.vocab_size

        # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
        # https://arxiv.org/pdf/2104.04473.pdf).
        checkpoint_activations_factor = 3
        seq_len = args.max_seq_length
        flops_per_step = (
            24
            * checkpoint_activations_factor
            * batch_size
            * seq_len
            * num_layers
            * (hidden_size**2)
        ) * (
            1.0
            + (seq_len / (6.0 * hidden_size))
            + (vocab_size / (16.0 * num_layers * hidden_size))
        )
        tflops = flops_per_step / (elapsed_time_per_step * world_size * (10**12))
        return tflops
