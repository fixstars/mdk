import argparse
import os

import huggingface_hub
import yaml
from huggingface_hub.constants import HF_TOKEN_PATH
from jinja2 import Environment, FileSystemLoader

YMLS = ["docker-compose.yml", "litellm-config.yml"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        default="config/launcher/llama3_8B_opt_350m.yaml",
        help="config yaml file",
    )
    parser.add_argument(
        "--output_dir",
        default="output/launcher",
        help="output directory",
    )
    parser.add_argument(
        "--container_basename",
        default="launcher",
        help="container basename",
    )
    args = parser.parse_args()

    # check existence
    docker_compose_yml_path = os.path.join(args.output_dir, "docker-compose.yml")
    if os.path.exists(docker_compose_yml_path):
        overwrite = input(f"{args.output_dir} already exists. Overwrite? (y/N): ")
        if overwrite.lower() not in ["y", "yes"]:
            print("Skipping generation.")
            return

    with open(f"{args.config_yaml}", "r") as f:
        docker_compose_litellm_configs = yaml.safe_load(f)
    docker_compose_litellm_configs["USER"] = os.environ["USER"]
    docker_compose_litellm_configs["CONTAINER_BASENAME"] = args.container_basename

    # bind Hugging Face token if available
    try:
        whoami = huggingface_hub.whoami()
        docker_compose_litellm_configs["HF_TOKEN_PATH"] = HF_TOKEN_PATH
        print(f"Logged in as {whoami['name']}")
    except huggingface_hub.errors.LocalTokenNotFoundError:
        # ログイン不要なモデルの場合は、特になにもしなくてもモデルが使えるので
        pass

    env = Environment(loader=FileSystemLoader("."), trim_blocks=True)
    os.makedirs(args.output_dir, exist_ok=True)
    for yml in YMLS:
        template = env.get_template(f"templates/docker/launcher-{yml}.jinja2")
        result = template.render(docker_compose_litellm_configs)
        with open(os.path.join(args.output_dir, yml), "w") as f:
            f.write(f"{result}\n")

    print(f"Config files {YMLS} generated to {args.output_dir} successfully.")


if __name__ == "__main__":
    main()
