import logging
import os
import time

import hydra

from preprocess.dpo_convert import dpo_convert
from preprocess.sft_convert import sft_convert

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../config/dataset_converter", config_name="config"
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

    if args.llm.generate.temperature == 0:
        if args.llm.generate.num_return_sequences != 1:
            logger.info(
                "`llm.generate.temperature=0` is only available when `llm.generate.return_sequences=1`"
            )
            return

    if args.mode == "sft":
        sft_convert(args)
    elif args.mode == "dpo":
        dpo_convert(args)
    logger.info("execution time: %f sec", time.time() - start_time)


if __name__ == "__main__":
    main()
