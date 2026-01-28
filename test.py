import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

# Define a simple neural network
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define a transformer-like model with attention
class SimpleAttentionModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.seq_len = seq_len

        self.embedding = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, d_model)

    def forward(self, x):
        x = self.embedding(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.fc1(attn_output)
        x = self.fc2(x)
        return x


def example1_conv_net():
    """Example 1: Counting FLOPs for a convolutional network"""
    print("=" * 80)
    print("EXAMPLE 1: Convolutional Neural Network")
    print("=" * 80)

    model = SimpleConvNet()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Model architecture:\n{model}\n")

    # Count FLOPs during forward pass
    with FlopCounterMode(display=False) as flop_counter:
        output = model(input_tensor)

    print(f"\nOutput shape: {output.shape}")
    print(f"Total FLOPs: {flop_counter.get_total_flops():,}")

    # Get detailed breakdown
    flop_counts = flop_counter.get_flop_counts()
    print("\nDetailed FLOP counts by module:")
    for module_name, ops in flop_counts.items():
        print(f"\n{module_name}:")
        for op, count in ops.items():
            print(f"  {op}: {count:,}")


def example2_attention():
    """Example 2: Counting FLOPs for attention mechanisms"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Attention Model")
    print("=" * 80)

    d_model = 256
    batch_size = 8
    seq_len = 128

    model = SimpleAttentionModel(d_model=d_model, seq_len=seq_len)
    input_tensor = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Model parameters:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {model.nhead}")
    print(f"  sequence_length: {seq_len}\n")

    with FlopCounterMode(display=False, depth=3) as flop_counter:
        output = model(input_tensor)

    print(f"\nOutput shape: {output.shape}")
    print(f"Total FLOPs: {flop_counter.get_total_flops():,}")


def example3_simple_operations():
    """Example 3: Counting FLOPs for basic operations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Basic Matrix Operations")
    print("=" * 80)

    # Matrix multiplication
    print("\n--- Matrix Multiplication ---")
    A = torch.randn(100, 200)
    B = torch.randn(200, 300)

    with FlopCounterMode(display=False) as flop_counter:
        C = torch.mm(A, B)

    print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
    print(f"Expected FLOPs: {100 * 300 * 2 * 200:,}")
    print(f"Counted FLOPs: {flop_counter.get_total_flops():,}")

    # Batch matrix multiplication
    print("\n--- Batch Matrix Multiplication ---")
    A_batch = torch.randn(10, 50, 60)
    B_batch = torch.randn(10, 60, 70)

    with FlopCounterMode(display=False) as flop_counter:
        C_batch = torch.bmm(A_batch, B_batch)

    print(f"A shape: {A_batch.shape}, B shape: {B_batch.shape}, C shape: {C_batch.shape}")
    print(f"Expected FLOPs: {10 * 50 * 70 * 2 * 60:,}")
    print(f"Counted FLOPs: {flop_counter.get_total_flops():,}")


def example4_backward_pass():
    """Example 4: Counting FLOPs including backward pass"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Forward and Backward Pass")
    print("=" * 80)

    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    input_tensor = torch.randn(32, 100, requires_grad=True)
    target = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()

    # Forward pass only
    print("\n--- Forward Pass Only ---")
    with FlopCounterMode(display=False) as flop_counter:
        output = model(input_tensor)
        loss = criterion(output, target)

    forward_flops = flop_counter.get_total_flops()
    print(f"Forward FLOPs: {forward_flops:,}")

    # Forward + Backward pass
    print("\n--- Forward + Backward Pass ---")
    with FlopCounterMode(display=False) as flop_counter:
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()

    total_flops = flop_counter.get_total_flops()
    print(f"Total FLOPs (forward + backward): {total_flops:,}")
    print(f"Backward FLOPs: {total_flops - forward_flops:,}")


def example5_custom_flop_formula():
    """Example 5: Register custom FLOP counting formula"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Custom FLOP Formula")
    print("=" * 80)

    from torch.utils.flop_counter import register_flop_formula

    # Register a custom operation
    @register_flop_formula(torch.ops.aten.add)
    def add_flop(a_shape, b_shape, out_shape=None, **kwargs):
        """Count element-wise addition as 1 FLOP per element"""
        from math import prod
        return prod(out_shape)

    # Now additions will be counted
    print("\n--- Custom Formula for Addition ---")
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)

    with FlopCounterMode(display=False) as flop_counter:
        z = torch.add(x, y)
        w = torch.mm(x, y)  # Also includes matmul for comparison

    print(f"\nTotal FLOPs: {flop_counter.get_total_flops():,}")
    flop_counts = flop_counter.get_flop_counts()
    for module_name, ops in flop_counts.items():
        for op, count in ops.items():
            print(f"{op}: {count:,}")


def example6_comparison():
    """Example 6: Compare different model architectures"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Comparing Different Architectures")
    print("=" * 80)

    models = {
        "Small": nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ),
        "Medium": nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ),
        "Large": nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    }

    input_tensor = torch.randn(32, 100)

    print("\nModel Complexity Comparison:")
    print("-" * 60)
    print(f"{'Model':<10} {'Parameters':<15} {'FLOPs':<15} {'Relative FLOPs'}")
    print("-" * 60)

    results = {}
    for name, model in models.items():
        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        # Count FLOPs
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(input_tensor)
            flops = flop_counter.get_total_flops()

        results[name] = {'params': params, 'flops': flops}

    base_flops = results["Small"]["flops"]

    for name, data in results.items():
        relative = data['flops'] / base_flops
        print(f"{name:<10} {data['params']:<15,} {data['flops']:<15,} {relative:.2f}x")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PyTorch FLOP Counter Examples" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")

    # Run all examples
    example1_conv_net()
    example2_attention()
    example3_simple_operations()
    example4_backward_pass()
    example5_custom_flop_formula()
    example6_comparison()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
