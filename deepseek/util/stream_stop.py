import sys
import time
from transformers import StoppingCriteria


class StreamStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_timestamps = []
        self.start_time = None

    def __call__(self, input_ids, scores, **kwargs):
        current_time = time.time()

        if self.start_time is None:
            self.start_time = current_time

        self.token_timestamps.append(current_time)

        # Print latest token (streaming behavior)
        new_token = self.tokenizer.decode(input_ids[0, -1], skip_special_tokens=False)
        sys.stdout.write(new_token)
        sys.stdout.flush()

        return False  # Continue generating tokens

    def compute_tps(self):
        """Calculate average tokens per second after generation."""
        if len(self.token_timestamps) < 2:
            return 0.0  # Avoid division by zero if only one token is generated

        total_tokens = len(self.token_timestamps)
        total_time = self.token_timestamps[-1] - self.token_timestamps[0]

        total_tokens = len(self.token_timestamps)
        return ((total_tokens / total_time if total_time > 0 else 0.0), total_tokens)
