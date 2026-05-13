"""
Defines the PromptIR model architecture and its constituent blocks.

This includes:
- Downsample and Upsample utility classes.
- PromptGenBlock: Generates dynamic spatial prompts.
- PromptInteractionBlock: Integrates prompts with decoder features.
- PromptIR: The main U-Net based model with Transformer blocks and
  the dynamic prompting mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# transformer_block.py is kept beside this file in the project root.
from transformer_block import TransformerBlock


class Downsample(nn.Module):
    """
    Downsampling block using a strided convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, H/2, W/2].
        """
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling block using PixelShuffle.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Conv layer outputs 4x channels for PixelShuffle with scale_factor=2
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.PixelShuffle(2)  # Upscales H and W by 2

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, H*2, W*2].
        """
        x = self.conv(x)
        x = self.upsample(x)
        return x


class DegradationClassifier(nn.Module):
    """
    Lightweight auxiliary classifier for rain/snow degradation prediction.
    """
    def __init__(self, in_channels, hidden_dim=128, num_classes=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(x))


class FrequencyGuidedBranch(nn.Module):
    """
    Extracts lightweight high-frequency cues from the degraded input image.
    """
    def __init__(self, out_channels):
        super().__init__()
        kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel.repeat(3, 1, 1, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        high_freq = F.conv2d(x, self.laplacian_kernel, padding=1, groups=3)
        return self.proj(high_freq)


class PromptGenBlock(nn.Module):
    """
    Generates a dynamic spatial prompt based on input decoder features.

    The prompt is a weighted sum of learnable spatial prompt components,
    where weights are dynamically predicted from the decoder features.
    The resulting prompt is interpolated to match the decoder feature map size.
    """
    def __init__(self, features_input_dim, num_prompt_components,
                 prompt_channel_dim, base_prompt_hw, bias=False,
                 use_task_prompt_bank=False, task_prompt_scale=0.5):
        """
        Args:
            features_input_dim (int): Channel dimension of the input decoder features.
            num_prompt_components (int): Number of learnable spatial prompt components.
            prompt_channel_dim (int): Channel dimension for each prompt component and the output prompt.
            base_prompt_hw (int or tuple): Initial (H, W) of the spatial prompt components.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        self.use_task_prompt_bank = use_task_prompt_bank
        self.task_prompt_scale = task_prompt_scale
        if isinstance(base_prompt_hw, int):
            base_prompt_h, base_prompt_w = base_prompt_hw, base_prompt_hw
        else:
            base_prompt_h, base_prompt_w = base_prompt_hw

        # Learnable spatial prompt components
        prompt_shape = (1, num_prompt_components, prompt_channel_dim, base_prompt_h, base_prompt_w)
        if use_task_prompt_bank:
            self.shared_prompt_components = nn.Parameter(torch.randn(*prompt_shape))
            self.rain_prompt_components = nn.Parameter(torch.randn(*prompt_shape))
            self.snow_prompt_components = nn.Parameter(torch.randn(*prompt_shape))
        else:
            self.prompt_components = nn.Parameter(torch.randn(*prompt_shape))
        # Linear layer to generate weights for combining prompt components
        self.weight_generator = nn.Linear(features_input_dim, num_prompt_components)
        # Final convolution to refine the generated prompt
        self.final_conv = nn.Conv2d(
            prompt_channel_dim, prompt_channel_dim, kernel_size=3, padding=1, bias=bias
        )
        self.last_prompt_weights = None

    def _weighted_sum(self, prompt_components, prompt_weights):
        b = prompt_weights.shape[0]
        weighted_prompts = prompt_weights.view(b, -1, 1, 1, 1) * prompt_components
        return torch.sum(weighted_prompts, dim=1)

    def forward(self, decoder_features, degradation_probs=None):
        """
        Dynamically generates a spatial prompt.
        Args:
            decoder_features (torch.Tensor): Features from a decoder stage [B, C_feat, H, W].
        Returns:
            torch.Tensor: Generated spatial prompt [B, C_prompt, H, W].
        """
        b, _, h, w = decoder_features.shape  # Use _ for c_feat as it's defined by features_input_dim

        # Generate weights for prompt components
        pooled_features = F.adaptive_avg_pool2d(decoder_features, (1, 1)).view(b, -1)  # [B, C_feat]
        prompt_weights = F.softmax(self.weight_generator(pooled_features), dim=1)  # [B, num_components]
        self.last_prompt_weights = prompt_weights.detach()

        if self.use_task_prompt_bank:
            shared_prompt = self._weighted_sum(self.shared_prompt_components, prompt_weights)
            rain_prompt = self._weighted_sum(self.rain_prompt_components, prompt_weights)
            snow_prompt = self._weighted_sum(self.snow_prompt_components, prompt_weights)

            if degradation_probs is None:
                degradation_probs = decoder_features.new_full((b, 2), 0.5)
            p_rain = degradation_probs[:, 0].view(b, 1, 1, 1)
            p_snow = degradation_probs[:, 1].view(b, 1, 1, 1)
            summed_prompt = shared_prompt + self.task_prompt_scale * (
                p_rain * rain_prompt + p_snow * snow_prompt
            )
        else:
            # Weighted sum of spatial prompt components
            # prompt_weights: [B, N] -> [B, N, 1, 1, 1] for broadcasting
            # self.prompt_components: [1, N, C_prompt, H_base, W_base]
            summed_prompt = self._weighted_sum(self.prompt_components, prompt_weights)

        # Interpolate to match decoder feature map size and refine
        interpolated_prompt = F.interpolate(summed_prompt, size=(h, w), mode='bilinear', align_corners=False)
        output_prompt = self.final_conv(interpolated_prompt)

        return output_prompt


class PromptInteractionBlock(nn.Module):
    """
    Integrates a generated prompt with decoder features using a TransformerBlock.
    """
    def __init__(self, feature_dim, prompt_dim, num_transformer_heads,
                 ffn_expansion_factor, bias):
        """
        Args:
            feature_dim (int): Channel dimension of the decoder features.
            prompt_dim (int): Channel dimension of the generated prompt.
            num_transformer_heads (int): Number of attention heads for the TransformerBlock.
            ffn_expansion_factor (float): Expansion factor for the FFN in TransformerBlock.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        concat_dim = feature_dim + prompt_dim
        self.transformer = TransformerBlock(
            dim=concat_dim, num_heads=num_transformer_heads,
            ffn_expansion_factor=ffn_expansion_factor, bias=bias
        )
        # Adjust channel dimension back to original feature_dim
        self.channel_adjust_conv = nn.Conv2d(concat_dim, feature_dim, kernel_size=1, bias=bias)

    def forward(self, features, prompt):
        """
        Args:
            features (torch.Tensor): Decoder features [B, C_feat, H, W].
            prompt (torch.Tensor): Generated spatial prompt [B, C_prompt, H, W].
        Returns:
            torch.Tensor: Features after interaction with prompt [B, C_feat, H, W].
        """
        # Concatenate features and prompt along channel dimension
        x = torch.cat([features, prompt], dim=1)
        x = self.transformer(x)
        x = self.channel_adjust_conv(x)
        return x + features  # Residual connection with original features


class PromptIR(nn.Module):
    """
    PromptIR model architecture.

    A U-Net based architecture with a Transformer backbone. It incorporates
    dynamic spatial prompting at multiple decoder stages for adaptive image restoration.
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 base_dim=48,
                 num_blocks_per_level=(4, 6, 6, 8), # Default from a successful experiment
                 num_refinement_blocks=4,
                 num_prompt_components=5,
                 pg_prompt_dim_map=(64, 128, 256), # C_prompt for PGM (deep, mid, shallow decoder stages)
                 pg_base_hw_map=(16, 32, 64),    # Base H,W for PGM components (deep, mid, shallow)
                 backbone_num_attn_heads=8,
                 prompt_interaction_num_attn_heads=8,
                 ffn_expansion_factor=2.66,
                 use_degradation_classifier=False,
                 use_task_prompt_bank=False,
                 use_frequency_branch=False,
                 task_prompt_scale=0.5,
                 bias=False):
        super().__init__()

        self.base_dim = base_dim
        self.num_levels = len(num_blocks_per_level)
        self.num_encoder_levels = self.num_levels - 1
        self.use_degradation_classifier = use_degradation_classifier
        self.use_task_prompt_bank = use_task_prompt_bank
        self.use_frequency_branch = use_frequency_branch

        if isinstance(backbone_num_attn_heads, int):
            backbone_num_attn_heads = [backbone_num_attn_heads] * self.num_levels

        # Initial convolution: projects input to base_dim
        self.initial_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=bias)

        # --- Encoder ---
        self.encoder_levels = nn.ModuleList()
        self.encoder_skip_dims = []  # To store dimensions for skip connections
        current_dim = base_dim
        for i in range(self.num_encoder_levels):
            self.encoder_skip_dims.append(current_dim)
            # Transformer blocks for the current encoder level
            level_blocks = nn.Sequential(*[
                TransformerBlock(dim=current_dim, num_heads=backbone_num_attn_heads[i],
                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_blocks_per_level[i])
            ])
            self.encoder_levels.append(level_blocks)
            self.encoder_levels.append(Downsample(current_dim, current_dim * 2))
            current_dim *= 2
        
        # --- Latent/Bottleneck ---
        # Transformer blocks at the bottleneck of the U-Net
        self.latent_transformers = nn.Sequential(*[
            TransformerBlock(dim=current_dim, num_heads=backbone_num_attn_heads[-1], # Use heads for deepest level
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks_per_level[-1]) # Last element is used for latent only
        ])
        bottleneck_dim = current_dim

        if self.use_degradation_classifier:
            self.degradation_classifier = DegradationClassifier(in_channels=bottleneck_dim)
        
        # --- Decoder with Prompt Blocks ---
        self.decoder_levels = nn.ModuleList()
        self.prompt_gens = nn.ModuleList()
        self.prompt_interactions = nn.ModuleList()

        # Decoder stages mirror the encoder skip levels.
        for i in range(self.num_encoder_levels):
            # Upsampling block
            upsample_out_dim = current_dim // 2
            self.decoder_levels.append(Upsample(current_dim, upsample_out_dim))
            
            # Concatenation with skip connection from corresponding encoder level
            skip_dim_idx = self.num_encoder_levels - 1 - i  # Skips are indexed from shallowest encoder level
            merged_dim_after_skip = upsample_out_dim + self.encoder_skip_dims[skip_dim_idx]
            
            # Convolution to merge channels to the target dimension for this decoder level
            target_current_dim_decoder = upsample_out_dim
            self.decoder_levels.append(
                nn.Conv2d(merged_dim_after_skip, target_current_dim_decoder, kernel_size=1, bias=bias)
            )
            current_dim = target_current_dim_decoder  # This is the main feature dimension for this decoder stage

            # Transformer blocks for this decoder level
            num_dec_tf_blocks = num_blocks_per_level[skip_dim_idx]
            dec_tf_heads = backbone_num_attn_heads[skip_dim_idx]
            decoder_transformer_stage = nn.Sequential(*[
                TransformerBlock(dim=current_dim, num_heads=dec_tf_heads,
                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_dec_tf_blocks)
            ])
            self.decoder_levels.append(decoder_transformer_stage)

            # Prompt Generation and Interaction for this stage
            # pg_prompt_dim_map/pg_base_hw_map keys 0,1,2 correspond to i=0,1,2 (deep to shallow prompt blocks)
            current_pg_prompt_channel_dim = pg_prompt_dim_map[i]
            current_pg_base_hw = pg_base_hw_map[i]

            self.prompt_gens.append(
                PromptGenBlock(features_input_dim=current_dim, # Input from decoder TFs
                                 num_prompt_components=num_prompt_components,
                                 prompt_channel_dim=current_pg_prompt_channel_dim,
                                 base_prompt_hw=current_pg_base_hw,
                                 use_task_prompt_bank=use_task_prompt_bank,
                                 task_prompt_scale=task_prompt_scale,
                                 bias=bias)
            )
            self.prompt_interactions.append(
                PromptInteractionBlock(feature_dim=current_dim, # Decoder features
                                         prompt_dim=current_pg_prompt_channel_dim, # PGM output dim
                                         num_transformer_heads=prompt_interaction_num_attn_heads,
                                         ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias)
            )

        # --- Refinement Stage ---
        # Transformer blocks operating on the output of the shallowest decoder stage
        # current_dim at this point should be equal to base_dim
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=base_dim, num_heads=backbone_num_attn_heads[0], # Use heads for shallowest level
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_refinement_blocks)
        ])

        if self.use_frequency_branch:
            self.frequency_branch = FrequencyGuidedBranch(out_channels=base_dim)
            self.frequency_fusion = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1, bias=bias)
        
        # Final convolution to produce the output image
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x_inp, return_aux=False):
        """
        Forward pass of the PromptIR model.
        Args:
            x_inp (torch.Tensor): Input degraded image tensor [B, C_in, H, W].
        Returns:
            torch.Tensor: Restored image tensor [B, C_out, H, W].
        """
        skip_connections = []
        
        # --- Encoder ---
        x = self.initial_conv(x_inp)
        enc_module_idx = 0
        for i in range(self.num_encoder_levels):
            x = self.encoder_levels[enc_module_idx](x)  # Apply Transformer blocks
            enc_module_idx += 1
            skip_connections.append(x) # Save for skip connection
            x = self.encoder_levels[enc_module_idx](x)  # Apply Downsample
            enc_module_idx += 1
        
        # --- Bottleneck ---
        x = self.latent_transformers(x)
        degradation_logits = None
        degradation_probs = None
        if self.use_degradation_classifier:
            degradation_logits = self.degradation_classifier(x)
            degradation_probs = F.softmax(degradation_logits, dim=1)

        # --- Decoder ---
        dec_module_idx = 0
        for i in range(self.num_encoder_levels): # Iterate through decoder stages
            x = self.decoder_levels[dec_module_idx](x)  # Upsample
            dec_module_idx += 1
            
            skip = skip_connections.pop()  # Get skip connection (from last to first stored)
            x = torch.cat([x, skip], dim=1) # Concatenate
            x = self.decoder_levels[dec_module_idx](x)  # Merge convolution
            dec_module_idx += 1
            
            # Main decoder Transformer blocks for this level
            x_after_tfs = self.decoder_levels[dec_module_idx](x)
            dec_module_idx += 1
            
            # Prompt Generation and Interaction
            generated_prompt = self.prompt_gens[i](x_after_tfs, degradation_probs=degradation_probs)
            x = self.prompt_interactions[i](x_after_tfs, generated_prompt)
            
        # --- Refinement & Final Output ---
        if self.use_frequency_branch:
            freq_feat = self.frequency_branch(x_inp)
            if freq_feat.shape[-2:] != x.shape[-2:]:
                freq_feat = F.interpolate(freq_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = self.frequency_fusion(torch.cat([x, freq_feat], dim=1))

        x = self.refinement(x)
        x_restored = self.final_conv(x)
        
        restored = x_restored + x_inp # Global residual connection
        if return_aux:
            return {
                "restored": restored,
                "degradation_logits": degradation_logits,
            }
        return restored


if __name__ == '__main__':
    # Example usage and test for the PromptIR model
    bs = 2  # Batch size
    img_size = 256  # Image dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n--- Testing PromptIR Model ---")
    # Model parameters should match those used in training/prediction scripts
    model = PromptIR(
        base_dim=48,
        num_blocks_per_level=[3, 4, 4, 6], # Example, adjust to match your config
        num_refinement_blocks=4,
        backbone_num_attn_heads=8,
        prompt_interaction_num_attn_heads=8,
        pg_prompt_dim_map={0: 256, 1: 128, 2: 64}, # Corresponds to 3 decoder prompt stages
        pg_base_hw_map={
            0: img_size // 16, # Deepest prompt stage
            1: img_size // 8,  # Middle prompt stage
            2: img_size // 4   # Shallowest prompt stage
        }
    ).to(device)

    test_input = torch.randn(bs, 3, img_size, img_size).to(device)
    print(f"Input shape: {test_input.shape}")
    
    try:
        output = model(test_input)
        print(f"PromptIR output shape: {output.shape}")
        assert output.shape == (bs, 3, img_size, img_size), "Output shape mismatch!"
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters for PromptIR: {total_params / 1e6:.2f} M")
        print("PromptIR forward pass successful.")
        
    except Exception as e:
        print(f"Error during PromptIR forward pass: {e}")
        import traceback
        traceback.print_exc()
