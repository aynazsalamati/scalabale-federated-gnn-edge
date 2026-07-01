import random
import torch


class AdaptiveScheduler:
    def __init__(self, threshold=0.002, availability=0.85, min_send_probability=0.45, verbose=False):
        self.threshold = threshold
        self.availability = availability
        self.min_send_probability = min_send_probability
        self.verbose = verbose

    def should_send_update(self, old_weights, new_weights):
        if random.random() > self.availability:
            return False

        total_change, total_norm = 0.0, 0.0
        for key in old_weights:
            total_change += torch.norm(new_weights[key].float() - old_weights[key].float()).item()
            total_norm += torch.norm(old_weights[key].float()).item()

        relative_change = total_change / (total_norm + 1e-12)
        if relative_change > self.threshold:
            return True
        return random.random() < self.min_send_probability
