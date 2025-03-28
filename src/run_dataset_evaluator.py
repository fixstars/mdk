import logging
import os
import time

import hydra

from preprocess.evaluate import evaluate
from preprocess.prompt_tuner import prompt_generate

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../config/dataset_evaluator",
    config_name="karakuri_apm",
)
def main(args):
    args.rank = int(os.getenv("LOCAL_RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))
    if args.rank > 0:
        logging.disable(logging.FATAL)

    start_time = time.time()
    if args.output.basedir is None:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        args.output.basedir = hydra_cfg.runtime.output_dir
    args.inputs.name = hydra.utils.to_absolute_path(args.inputs.name)
    logger.debug("args: %s", args)

    if args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "prompt_generate":
        prompt_generate(args)

    logger.info("execution time: %f sec", time.time() - start_time)


if __name__ == "__main__":
    main()
