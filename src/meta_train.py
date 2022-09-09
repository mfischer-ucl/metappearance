import os
import hydra
from metappearance import Metappearance
from omegaconf import OmegaConf, DictConfig, open_dict
from pytorch_lightning.utilities.seed import seed_everything
from utils.utils import resolve_data, resolve_model, resolve_loss

seed_everything(123)


@hydra.main(config_path="../config/", config_name="main")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    seed_everything(43)

    # load application-specific config file
    subdict = OmegaConf.load(os.path.join(cfg.general.basepath,
                                          'config', '{}.yaml'.format(cfg.general.mode)))

    # add application-specific cfg details to main cfg
    with open_dict(cfg):
        cfg.data = subdict.data
        cfg.train = subdict.train
        cfg.meta = subdict.meta
        cfg.model = subdict.model
        subdict.device = cfg.device
        subdict.basepath = cfg.general.basepath

    # get train- and test-task-distributions
    trainDistr, testDistr = resolve_data(mode=cfg.general.mode,
                                         basepath=cfg.general.basepath,
                                         appl_subdict=subdict)

    # construct inner-loop model
    model = resolve_model(mode=cfg.general.mode,
                          appl_subdict=subdict,
                          verbose=True).to(cfg.device)

    # load corresponding loss function, e.g., VGG stats for texture, LogMAE for BRDF, ...
    lossfn = resolve_loss(mode=cfg.general.mode,
                          appl_subdict=subdict)

    # construct meta-class
    meta_model = Metappearance(cfg,
                               model=model,
                               lossfn=lossfn,
                               trainDistr=trainDistr,
                               testDistr=testDistr)

    # train
    meta_model.execute_outer_loop()


if __name__ == '__main__':
    main()

