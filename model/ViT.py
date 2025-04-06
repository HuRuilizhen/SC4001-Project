import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=256) -> None:
        """
        Initializes the PatchEmbedding layer for the Vision Transformer.

        Parameters:
        -----------
        img_size : int, optional
            The size of the input image (height and width), by default 28.
        patch_size : int, optional
            The size of each patch (height and width), by default 4.
        in_channels : int, optional
            The number of input channels (e.g., 1 for grayscale images), by default 1.
        embed_dim : int, optional
            The dimensionality of the embedding space, by default 256.
        """
        super().__init__()

        self.img_size = img_size

        self.patch_size = patch_size

        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the input image to the embedding space.

        Parameters:
        -----------
        x : torch.Tensor
            The input image tensor with shape (batch_size, channels, height, width)

        Returns:
        -------
        torch.Tensor
            The projected tensor with shape (batch_size, n_patches, embed_dim)
        """
        # x shape = [b_ssize, in_chan, i_size, i_size]
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        # final shape = [b_ssize, n_patch, embed_dm]
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_channels=1,
        n_classes=10,
        embed_dim=256,
        depth=4,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ) -> None:
        """
        Initializes the Vision Transformer model.

        Parameters:
        -----------
        img_size : int, optional
            The size of the input image (height and width), by default 28.
        patch_size : int, optional
            The size of each patch (height and width), by default 4.
        in_channels : int, optional
            The number of input channels (e.g., 1 for grayscale images), by default 1.
        n_classes : int, optional
            The number of output classes for classification, by default 10.
        embed_dim : int, optional
            The dimensionality of the embedding space, by default 256.
        depth : int, optional
            The number of transformer layers, by default 4.
        n_heads : int, optional
            The number of attention heads in the multi-head attention mechanism, by default 8.
        mlp_ratio : float, optional
            The ratio of the hidden dimension in the MLP block to the embedding dimension, by default 4.0.
        dropout : float, optional
            The dropout rate used in the encoder layers, by default 0.1.
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        n_patches = self.patch_embed.n_patches

        # class  token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # pos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # layer norm
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, n_classes)

        # pos embedding initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Vision Transformer.

        Parameters:
        -----------
            x : torch.Tensor
                The input tensor with shape (batch_size, channels, height, width).

        Returns:
        -------
            The output tensor with shape (batch_size, n_classes).
        """

        # x shape=[b_size, in_channels, img_size, img_size]
        b_size = x.shape[0]

        x = self.patch_embed(x)

        #   class token
        cls_tokens = self.cls_token.expand(b_size, -1, -1)  # [b_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b_size, n_patches + 1, embed_dim]

        # add embeddings
        x = x + self.pos_embed

        # apply transformer
        x = self.transformer(x)

        # apply normalization
        x = self.norm(x)

        # classification from the first (class) token
        x = x[:, 0]
        x = self.head(x)

        return x
