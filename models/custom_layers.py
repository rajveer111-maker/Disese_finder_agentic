import tensorflow as tf

from typing import Dict


_DEPTHWISE_LAYER_NAME_MAP: Dict[str, Dict[str, str]] = {
    "depthwise_separable_conv_block_3": {
        "depthwise": "depthwise_conv2d_3",
        "bn_depth": "batch_normalization_10",
        "pointwise": "conv2d_15",
        "bn_point": "batch_normalization_11",
    },
    "depthwise_separable_conv_block_4": {
        "depthwise": "depthwise_conv2d_4",
        "bn_depth": "batch_normalization_13",
        "pointwise": "conv2d_21",
        "bn_point": "batch_normalization_14",
    },
    "depthwise_separable_conv_block_5": {
        "depthwise": "depthwise_conv2d_5",
        "bn_depth": "batch_normalization_16",
        "pointwise": "conv2d_27",
        "bn_point": "batch_normalization_17",
    },
}

_MULTISCALE_LAYER_NAME_MAP: Dict[str, Dict[str, str]] = {
    "multi_scale_feature_fusion_2": {
        "conv_reduce": "conv2d_16",
        "conv_3x3": "conv2d_17",
        "conv_5x5": "conv2d_18",
        "conv_1x1": "conv2d_19",
        "conv_fuse": "conv2d_20",
        "bn": "batch_normalization_12",
    },
    "multi_scale_feature_fusion_3": {
        "conv_reduce": "conv2d_22",
        "conv_3x3": "conv2d_23",
        "conv_5x5": "conv2d_24",
        "conv_1x1": "conv2d_25",
        "conv_fuse": "conv2d_26",
        "bn": "batch_normalization_15",
    },
}

_SQUEEZE_EXCITATION_NAME_MAP: Dict[str, Dict[str, str]] = {
    "squeeze_excitation_5": {"dense_1": "dense_11", "dense_2": "dense_12"},
    "squeeze_excitation_6": {"dense_1": "dense_13", "dense_2": "dense_14"},
    "squeeze_excitation_7": {"dense_1": "dense_15", "dense_2": "dense_16"},
    "squeeze_excitation_8": {"dense_1": "dense_17", "dense_2": "dense_18"},
    "squeeze_excitation_9": {"dense_1": "dense_19", "dense_2": "dense_20"},
}


def _get_layer_name(name_map: Dict[str, Dict[str, str]], layer_name: str, key: str, default_suffix: str) -> str:
    return name_map.get(layer_name, {}).get(key, f"{layer_name}_{default_suffix}")


@tf.keras.utils.register_keras_serializable(package="Agentic")
class DepthwiseSeparableConvBlock(tf.keras.layers.Layer):
    """Depthwise separable convolution block used by the Adaptive MSF network."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        depthwise_name = _get_layer_name(_DEPTHWISE_LAYER_NAME_MAP, self.name, "depthwise", "depthwise")
        bn_depth_name = _get_layer_name(_DEPTHWISE_LAYER_NAME_MAP, self.name, "bn_depth", "bn_depth")
        pointwise_name = _get_layer_name(_DEPTHWISE_LAYER_NAME_MAP, self.name, "pointwise", "pointwise")
        bn_point_name = _get_layer_name(_DEPTHWISE_LAYER_NAME_MAP, self.name, "bn_point", "bn_point")

        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            name=depthwise_name,
        )
        self.bn_depth = tf.keras.layers.BatchNormalization(name=bn_depth_name)
        self.pointwise = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            name=pointwise_name,
        )
        self.bn_point = tf.keras.layers.BatchNormalization(name=bn_point_name)
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        x = self.depthwise(inputs)
        x = self.bn_depth(x, training=training)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.bn_point(x, training=training)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


@tf.keras.utils.register_keras_serializable(package="Agentic")
class MultiScaleFeatureFusion(tf.keras.layers.Layer):
    """Multi-scale feature fusion block combining multiple receptive fields."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        branch_filters = max(1, filters // 4)

        conv_reduce_name = _get_layer_name(_MULTISCALE_LAYER_NAME_MAP, self.name, "conv_reduce", "conv_reduce")
        conv_3x3_name = _get_layer_name(_MULTISCALE_LAYER_NAME_MAP, self.name, "conv_3x3", "conv_3x3")
        conv_5x5_name = _get_layer_name(_MULTISCALE_LAYER_NAME_MAP, self.name, "conv_5x5", "conv_5x5")
        conv_1x1_name = _get_layer_name(_MULTISCALE_LAYER_NAME_MAP, self.name, "conv_1x1", "conv_1x1")
        conv_fuse_name = _get_layer_name(_MULTISCALE_LAYER_NAME_MAP, self.name, "conv_fuse", "conv_fuse")
        bn_name = _get_layer_name(_MULTISCALE_LAYER_NAME_MAP, self.name, "bn", "bn")

        self.conv_reduce = tf.keras.layers.Conv2D(
            branch_filters,
            kernel_size=(1, 1),
            padding="same",
            activation="relu",
            name=conv_reduce_name,
        )
        self.conv_3x3 = tf.keras.layers.Conv2D(
            branch_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name=conv_3x3_name,
        )
        self.conv_5x5 = tf.keras.layers.Conv2D(
            branch_filters,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
            name=conv_5x5_name,
        )
        self.conv_1x1 = tf.keras.layers.Conv2D(
            branch_filters,
            kernel_size=(1, 1),
            padding="same",
            activation="relu",
            name=conv_1x1_name,
        )
        self.conv_fuse = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            name=conv_fuse_name,
        )
        self.bn = tf.keras.layers.BatchNormalization(name=bn_name)
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        branch1 = self.conv_reduce(inputs)
        branch2 = self.conv_3x3(inputs)
        branch3 = self.conv_5x5(inputs)
        branch4 = self.conv_1x1(inputs)

        x = tf.concat([branch1, branch2, branch3, branch4], axis=-1)
        x = self.conv_fuse(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


@tf.keras.utils.register_keras_serializable(package="Agentic")
class SqueezeExcitation(tf.keras.layers.Layer):
    """Squeeze-and-Excitation attention block."""

    def __init__(self, filters: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = max(1, reduction)

        dense_names = _SQUEEZE_EXCITATION_NAME_MAP.get(self.name, {})
        dense1_name = dense_names.get("dense_1", f"{self.name}_dense_1")
        dense2_name = dense_names.get("dense_2", f"{self.name}_dense_2")

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(
            max(1, filters // self.reduction),
            activation="relu",
            name=dense1_name,
        )
        self.dense2 = tf.keras.layers.Dense(
            filters,
            activation="sigmoid",
            name=dense2_name,
        )

    def call(self, inputs, training=None):
        x = self.global_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "reduction": self.reduction})
        return config

