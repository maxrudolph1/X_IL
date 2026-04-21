import logging
import random
import os
import re
import hydra
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _cfg_select(cfg, key, default=None):
    value = OmegaConf.select(cfg, key, default=default)
    if value in ("", "null", "None"):
        return default
    return value


def _sanitize_wandb_name(value):
    if value is None:
        return None
    name = str(value).replace("_", "-").replace("/", "-").lower()
    name = re.sub(r"[^a-z0-9.+-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name or None


def _hydra_choice(name):
    try:
        return HydraConfig.get().runtime.choices.get(name)
    except Exception:
        return None


def _infer_agent_name(cfg):
    agent_choice = _hydra_choice("agents")
    if agent_choice:
        agent_name = os.path.basename(str(agent_choice))
        if agent_name.endswith("_agent"):
            agent_name = agent_name[: -len("_agent")]
        return _sanitize_wandb_name(agent_name)

    agent_target = _cfg_select(cfg, "agents._target_")
    if agent_target:
        target_name = str(agent_target).split(".")[-1]
        target_name = re.sub(r"_?agent$", "", target_name, flags=re.IGNORECASE)
        return _sanitize_wandb_name(target_name)

    return None


def _infer_model_parts(agent_name):
    model_choice = _hydra_choice("agents/model")
    if not model_choice:
        return []

    model_name = os.path.splitext(os.path.basename(str(model_choice)))[0]
    tokens = [token for token in re.split(r"[_-]+", model_name.lower()) if token]
    known_agents = {"awr", "bc", "beso", "ddpm", "fm"}
    if tokens and (tokens[0] == agent_name or tokens[0] in known_agents):
        tokens = tokens[1:]

    parts = []
    if tokens and tokens[0] in {"dec", "encdec"}:
        parts.append(tokens.pop(0))

    # Transformer is the default architecture in several scripts. Omitting it
    # keeps the common names short, e.g. bc-dec-seed0.
    if tokens != ["transformer"]:
        parts.extend(tokens)

    return [_sanitize_wandb_name(part) for part in parts if _sanitize_wandb_name(part)]


def resolve_wandb_names(cfg):
    explicit_group = _cfg_select(cfg, "run_group") or _cfg_select(cfg, "wandb.group")
    explicit_name = _cfg_select(cfg, "run_name") or _cfg_select(cfg, "wandb.name")

    agent_name = _infer_agent_name(cfg)
    model_parts = _infer_model_parts(agent_name)
    inferred_group = "-".join([part for part in [agent_name] + model_parts if part])

    run_group = (
        _sanitize_wandb_name(explicit_group)
        or _sanitize_wandb_name(inferred_group)
        or _sanitize_wandb_name(_cfg_select(cfg, "group"))
    )
    run_name = _sanitize_wandb_name(explicit_name)
    if run_name is None and run_group is not None:
        run_name = f"{run_group}-seed{cfg.seed}"

    return run_name, run_group


@hydra.main(config_path="configs", config_name="libero_config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)
    run_name, run_group = resolve_wandb_names(cfg)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.config["wandb_run_name"] = run_name
    wandb.config["wandb_run_group"] = run_group

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        group=run_group,
        # mode="disabled",
        config=wandb.config
    )

    # load vqvae before training the agent: add path to the config file
    # train the agent
    agent = hydra.utils.instantiate(cfg.agents)
    trainer = hydra.utils.instantiate(cfg.trainers)

    agent.get_params()
    trainer.main(agent)

    # # simulate the model
    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.get_task_embs(trainer.trainset.tasks)

    env_sim.test_agent(agent, cfg.agents, epoch=cfg.epoch)

    log.info("Training done")
    log.info("state_dict saved in {}".format(agent.working_dir))
    wandb.finish()


if __name__ == "__main__":
    main()
