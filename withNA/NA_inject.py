import torch
from functools import wraps


def inject(
    corruption_probability: float = 0.1,  # fraction of elements to corrupt
    nan_frac: float = 1.0,  # share of corruptions that become NaN
    posinf_frac: float = 0.0,  # … +∞
    neginf_frac: float = 0.0,  # … –∞
    generator: torch.Generator | None = None,
):
    """
    Inject corruption (NaN, +∞, -∞) into PyTorch tensors during function execution.

    Examples:
        # Basic usage with @ decorator syntax
        @inject(corruption_probability=0.1, nan_frac=1.0)
        def my_function(x):
            return x

        # Neural network with @ decorator syntax
        class MyModel(nn.Module):
            @inject(corruption_probability=0.2, nan_frac=0.5, posinf_frac=0.5)
            def forward(self, x):
                return self.linear(x)

        # Post-application to existing model
        model = MyModel()
        model.forward = inject(corruption_probability=0.15, nan_frac=1.0)(model.forward)

        # Mixed corruption types
        @inject(corruption_probability=0.3, nan_frac=1/3, posinf_frac=1/3, neginf_frac=1/3)
        def mixed_corruption(x):
            return x

    Parameters:
        corruption_probability (float): Fraction of elements to corrupt (default: 0.1)
        nan_frac (float): Share of corruptions that become NaN (default: 1.0)
        posinf_frac (float): Share of corruptions that become +∞ (default: 0.0)
        neginf_frac (float): Share of corruptions that become -∞ (default: 0.0)
        generator (torch.Generator): Random generator for reproducibility (default: None)

    Returns:
        Decorated function that corrupts input tensors before execution.

    Works on CPU & GPU, preserves autograd.
    """
    # normalize corruption mix
    sum_fractions = nan_frac + posinf_frac + neginf_frac
    if sum_fractions > 0:  # Avoid division by zero
        nan_frac, posinf_frac, neginf_frac = (
            nan_frac / sum_fractions,
            posinf_frac / sum_fractions,
            neginf_frac / sum_fractions,
        )

    def _decorator(forward_fn):
        @wraps(forward_fn)
        def _wrapper(*args, **kwargs):
            new_args = list(args)

            # if first arg is a tensor, start from 0, else start from 1
            start_index = 0 if (len(args) > 0 and torch.is_tensor(args[0])) else 1

            for i in range(start_index, len(new_args)):
                tensor = new_args[i]
                if not torch.is_tensor(tensor) or corruption_probability <= 0 or tensor.numel() == 0:
                    continue

                # Make a copy to ensure we can modify it
                corrupted_tensor = tensor.clone()
                num_elements = corrupted_tensor.numel()
                device = corrupted_tensor.device
                random_generator = generator

                corruption_mask = torch.rand(num_elements, device=device, generator=random_generator) < corruption_probability
                if not corruption_mask.any():
                    continue

                corrupted_indices = corruption_mask.nonzero(as_tuple=False).squeeze(1)
                corruption_choice = torch.rand(corrupted_indices.numel(), device=device, generator=random_generator)

                corruption_values = torch.empty_like(corruption_choice)
                corruption_values[corruption_choice < nan_frac] = torch.nan
                corruption_values[
                    (corruption_choice >= nan_frac) & (corruption_choice < nan_frac + posinf_frac)
                ] = torch.inf
                corruption_values[corruption_choice >= nan_frac + posinf_frac] = -torch.inf

                corrupted_tensor.view(-1)[corrupted_indices] = corruption_values
                new_args[i] = corrupted_tensor

            return forward_fn(*new_args, **kwargs)

        return _wrapper

    return _decorator

