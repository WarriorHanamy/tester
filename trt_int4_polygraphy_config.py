import tensorrt as trt


def postprocess_config(builder, network, config):
    # Polygraphy CLI does not expose --int4, so enable it via TensorRT API.
    config.set_flag(trt.BuilderFlag.INT4)
    return config
