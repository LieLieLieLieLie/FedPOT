from config import Config

def get_data_module(cfg: Config):
    if cfg.data.dataset == "office_caltech":
        from feddata.office_caltech import OfficeCaltechDataModule
        return OfficeCaltechDataModule(cfg)
    elif cfg.data.dataset == "cwru":
        from feddata.cwru import CWRUDataModule
        return CWRUDataModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")