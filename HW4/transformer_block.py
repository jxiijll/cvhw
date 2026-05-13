"""
Defines custom Transformer-related blocks for image restoration models,
including Layer Normalization, Multi-DConv Head Transposed Self-Attention (MDTA),
Gated-Dconv Feed-Forward Network (GDFN), and the main TransformerBlock
which incorporates a light local processing branch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer Normalization for 2D feature maps (image-like data).
    Normalizes across the channel dimension.
    """
    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim (int): Number of channels (features) to normalize.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))  # Learnable scale parameter
        self.beta = nn.Parameter(torch.zeros(dim)) # Learnable shift parameter
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        # Calculate mean and variance across the channel dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)

        # Normalize
        std = torch.sqrt(variance + self.eps)
        x_normalized = (x - mean) / std

        # Apply learnable scale and shift
        # Reshape gamma and beta to [1, C, 1, 1] for broadcasting
        x_scaled = x_normalized * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return x_scaled


class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Self-Attention (MDTA) block.
    Inspired by Restormer.
    """
    def __init__(self, dim, num_heads=8, bias=False):
        """
        Args:
            dim (int): Input and output channel dimension.
            num_heads (int): Number of attention heads.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        # Standard scaling factor for attention, replaces learnable temperature
        self.scale = self.dim_head ** -0.5

        # Convolution to generate Q, K, V projections
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # Depth-wise convolution for Q, K, V
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    padding=1, groups=dim * 3, bias=bias)
        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of the same shape after attention.
        """
        b, c, h, w = x.shape

        # Generate Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  # Each is [B, C, H, W]

        # Reshape for multi-head attention
        # [B, num_heads, C_head, H*W]
        q = q.reshape(b, self.num_heads, self.dim_head, h * w)
        k = k.reshape(b, self.num_heads, self.dim_head, h * w)
        v = v.reshape(b, self.num_heads, self.dim_head, h * w)

        # Normalize Q and K before dot product (L2 normalization along feature dimension)
        q = F.normalize(q, dim=-1) # Normalize along the C_head dimension (transposed view)
        k = F.normalize(k, dim=-1) # Normalize along the C_head dimension (transposed view)
                                    # Actually, this normalizes along the sequence_length (H*W) dim
                                    # For Restormer-style, normalization is typically on the C_head dim.
                                    # Let's assume the original intent was to normalize the feature vectors.
                                    # If q is [B, num_heads, C_head, N], normalize along C_head (dim=2)
                                    # q = F.normalize(q, dim=2) 
                                    # k = F.normalize(k, dim=2)
                                    # However, the original code normalized dim=-1 (N=H*W).
                                    # Keeping original normalization for now.

        # Attention mechanism
        # (q @ k.transpose(-2, -1)) -> [B, num_heads, C_head, N] @ [B, num_heads, N, C_head] -> [B, num_heads, C_head, C_head]
        # This seems to be attention over feature channels if C_head is sequence length.
        # Standard attention: (Q K^T / sqrt(d_k)) V
        # Here, Q, K, V are [B, num_heads, C_head, N=H*W]
        # attn = (q.transpose(-2, -1) @ k) * self.scale # [B, num_heads, N, N]
        # This is spatial attention. The original code did:
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, num_heads, C_head, C_head]
                                                      # This is channel attention if C_head is the feature dim per head.
        attn = attn.softmax(dim=-1)

        # Apply attention to V
        # out = (v @ attn.transpose(-2, -1)) # If attn is [B, num_heads, N, N]
        # Original:
        out = (attn @ v)  # [B, num_heads, C_head, N]
        
        # Reshape back to [B, C, H, W]
        out = out.reshape(b, c, h, w) # c = num_heads * C_head

        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN).
    Inspired by Restormer.
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        """
        Args:
            dim (int): Input and output channel dimension.
            ffn_expansion_factor (float): Factor to expand dim to hidden_features.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        # Input projection: expands channels and splits for gating
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # Depth-wise convolution on the expanded features
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        # Output projection: contracts channels back to original dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1) # Split into two paths for gating
        # Gated Linear Unit (GLU) like operation with GELU activation
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """
    Main Transformer Block combining MDTA and GDFN with LayerNorm.
    Includes a "Light Local Branch" prepended for enhanced local feature capture.
    """
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False):
        """
        Args:
            dim (int): Input and output channel dimension.
            num_heads (int): Number of attention heads for MDTA.
            ffn_expansion_factor (float): Expansion factor for GDFN.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        # Light Local Branch: captures local features
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias), # Depth-wise conv
            nn.GELU()
        )
        
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # Light Local Branch processing
        local_ft = self.local_branch(x)
        
        # Main Transformer path (Attention and FFN with pre-normalization)
        x_main = x + self.attn(self.norm1(x))  # Self-attention with residual
        x_main = x_main + self.ffn(self.norm2(x_main)) # Feed-forward with residual
        
        # Add output of local branch to the output of main transformer path
        out = x_main + local_ft
        return out


if __name__ == '__main__':
    # Example Usage and Tests
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing TransformerBlock components on device: {device}")

    # Test TransformerBlock
    dim_test = 64
    num_heads_test = 4
    transformer = TransformerBlock(dim=dim_test, num_heads=num_heads_test).to(device)
    sample_input_tb = torch.randn(2, dim_test, 32, 32).to(device) # Batch size 2
    try:
        output_tb = transformer(sample_input_tb)
        print(f"TransformerBlock input shape: {sample_input_tb.shape}")
        print(f"TransformerBlock output shape: {output_tb.shape}")
        assert output_tb.shape == sample_input_tb.shape, "TransformerBlock shape mismatch!"
        print("TransformerBlock test successful.")
    except Exception as e:
        print(f"Error in TransformerBlock test: {e}")

    # Test MDTA
    mdta = MDTA(dim=dim_test, num_heads=8).to(device)
    sample_input_mdta = torch.randn(2, dim_test, 16, 16).to(device)
    try:
        output_mdta = mdta(sample_input_mdta)
        print(f"MDTA input shape: {sample_input_mdta.shape}")
        print(f"MDTA output shape: {output_mdta.shape}")
        assert output_mdta.shape == sample_input_mdta.shape, "MDTA shape mismatch!"
        print("MDTA test successful.")
    except Exception as e:
        print(f"Error in MDTA test: {e}")


    # Test GDFN
    gdfn = GDFN(dim=dim_test).to(device)
    sample_input_gdfn = torch.randn(2, dim_test, 16, 16).to(device)
    try:
        output_gdfn = gdfn(sample_input_gdfn)
        print(f"GDFN input shape: {sample_input_gdfn.shape}")
        print(f"GDFN output shape: {output_gdfn.shape}")
        assert output_gdfn.shape == sample_input_gdfn.shape, "GDFN shape mismatch!"
        print("GDFN test successful.")
    except Exception as e:
        print(f"Error in GDFN test: {e}")

    # Test LayerNorm
    ln = LayerNorm(dim=dim_test).to(device)
    sample_input_ln = torch.randn(2, dim_test, 16, 16).to(device)
    try:
        output_ln = ln(sample_input_ln)
        print(f"LayerNorm input shape: {sample_input_ln.shape}")
        print(f"LayerNorm output shape: {output_ln.shape}")
        assert output_ln.shape == sample_input_ln.shape, "LayerNorm shape mismatch!"
        # Check if mean is close to 0 and std is close to 1 across channel dim
        # print(f"Mean after LayerNorm: {output_ln.mean(dim=1)}") # Should be close to 0
        # print(f"Std after LayerNorm: {output_ln.std(dim=1)}")   # Should be close to 1
        print("LayerNorm test successful.")
    except Exception as e:
        print(f"Error in LayerNorm test: {e}")
