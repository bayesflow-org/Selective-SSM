import keras

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba Wrapper is not available")

from bayesflow.networks.summary_network import SummaryNetwork


class MambaBlock(keras.layers.Layer):
    def __init__(
        self,
        state_dim: int, 
        conv_dim: int,
        feature_dim: int = 16,
        expand: int = 1, 
        dt_min=0.001,
        dt_max=0.1, 
        device: str = "cuda",
        **kwargs
    ):
        """
        A Keras layer implementing a Mamba-based sequence processing block.

        This layer applies a Mamba model for sequence modeling, preceded by a 
        convolutional projection and followed by layer normalization.

        Parameters
        ----------
        state_dim : int
            The dimension of the state space in the Mamba model.
        conv_dim : int
            The dimension of the convolutional layer used in Mamba.
        feature_dim : int, optional
            The feature dimension for input projection and Mamba processing (default is 16).
        expand : int, optional
            Expansion factor for Mamba's internal dimension (default is 1).
        dt_min : float, optional
            Minimum delta time for Mamba (default is 0.001).
        dt_max : float, optional
            Maximum delta time for Mamba (default is 0.1).
        device : str, optional
            The device to which the Mamba model is moved, typically "cuda" or "cpu" (default is "cuda").
        **kwargs : dict
            Additional keyword arguments passed to the `keras.layers.Layer` initializer.

        Returns
        -------
        Tensor
            The output tensor after applying input projection, Mamba, and residual layer normalization.

        Examples
        --------
        >>> import tensorflow as tf
        >>> from keras.models import Model
        >>> from keras.layers import Input
        >>> block = MambaBlock(state_dim=64, conv_dim=32, feature_dim=16)
        >>> x = keras.random.normal((8, 100, 16))  # (batch_size, sequence_length, input_dim)
        >>> y = block(x, training=True)
        >>> print(y.shape)
        (8, 100, 16)
        """
        super().__init__(**kwargs)

        self.mamba = Mamba(
            d_model=feature_dim,
            d_state=state_dim, 
            d_conv=conv_dim, 
            expand=expand, 
            dt_min=dt_min, 
            dt_max=dt_max
        ).to(device)

        self.input_projector = keras.layers.Conv1D(feature_dim, kernel_size=1, strides=1,)
        self.layer_norm = keras.layers.LayerNormalization()
        
    def call(self, x, training: bool = False, **kwargs):
        x = self.input_projector(x)
        h = self.mamba(x)
        out = self.layer_norm(h + x, training=training, **kwargs)
        return out


class MambaSSM(SummaryNetwork):
    def __init__(
        self,
        summary_dim: int = 16,
        feature_dim: int = 64,
        mamba_blocks: int = 2,
        state_dim: int = 128,
        conv_dim: int = 32,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cuda",
        **kwargs,
    ):
        """
        A time-series summarization network using Mamba-based State Space Models (SSM). This model processes 
        sequential input data using a sequence of Mamba SSM layers, followed by optional pooling, dropout, 
        and a dense layer for extracting summary statistics.

        Mamba2 support currently unabailble due to stability issues

        Parameters
        ----------
        feature_dim : int
            The dimensionality of the Mamba SSM model.
        summary_dim : int, optional
            The output dimensionality of the summary statistics layer (default is 8).
        mamba_blocks : int, optional
            The number of sequential mamba blocks to use (default is 2).
        state_dim : int, optional
            The dimensionality of the internal state representation (default is 16).
        conv_dim : int, optional
            The dimensionality of the convolutional layer in Mamba (default is 4).
        expand : int, optional
            The expansion factor for the hidden state in Mamba (default is 2).
        dt_min : float, optional
            Minimum dynamic state evolution over time (default is 0.001).
        dt_max : float, optional
            Maximum dynmaic state evolution over time (default is 0.1).
        pooling : bool, optional
            Whether to apply global average pooling (default is True).
        dropout : int, float, or None, optional
            Dropout rate applied before the summary layer (default is 0.5).
        device : str, optional
            The computing device. Currently, only "cuda" is supported (default is "cuda").
        **kwargs : dict
            Additional keyword arguments passed to the `SummaryNetwork` parent class.
        """

        super().__init__(**kwargs)
        if device != "cuda":
            raise NotImplementedError("MambaSSM currently only supports cuda")

        self.mamba_blocks = []
        for i in range(mamba_blocks):
            if i == 0:
                feature_dim = feature_dim
            else:
                feature_dim = state_dim
            self.mamba_blocks.append(MambaBlock(feature_dim, state_dim, conv_dim, expand, dt_min, dt_max, device))

        self.pooler = keras.layers.GlobalAveragePooling1D()
        self.summary_stats = keras.layers.Dense(summary_dim)

    def call(self, time_series, **kwargs):
        
        summary = time_series
        for mamba_block in self.mamba_blocks:
            summary = mamba_block(summary, **kwargs)
    
        summary = self.pooler(summary)
        summary = self.summary_stats(summary)

        return summary
