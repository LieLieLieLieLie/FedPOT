from config import Config

def get_data_module(cfg: Config):
    if cfg.data.dataset == "office_home":
        from feddata.office_home import OfficeHomeDataModule
        return OfficeHomeDataModule(cfg)
    elif cfg.data.dataset == "cwru":
        from feddata.cwru import CWRUDataModule
        return CWRUDataModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")
