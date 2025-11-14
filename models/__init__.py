def build_models(cfg):
    if cfg.NAME.startswith("InterMoE"):
        from .intermoe import InterMoE
        model = InterMoE(cfg)
    elif cfg.NAME.startswith("CasualSTVAE"):
        from .vae import CasualSTVAE
        model = CasualSTVAE(cfg)
    else:
        raise KeyError(f"Model {cfg.NAME} not recognized.")
    return model