#!/bin/bash

set -x
CONFIG_YAML=${1:-config/launcher/llama3_8B_opt_350m.yaml}
if [ ! -f "${CONFIG_YAML}" ]; then
  echo "Error: Config YAML '${CONFIG_YAML}' not found."
  exit 1
fi

# generate unique PROJECT_NAME to allow parallel launches
CONFIG_NAME=$(basename -- "$CONFIG_YAML")
RANDOM_STRING=$(tr -dc 'a-z' < /dev/urandom | fold -w 8 | head -n 1)
PROJECT_DIR="outputs/launcher/${CONFIG_NAME%.*}"

# generate yml
python src/launcher/generate_yml.py --config_yaml ${CONFIG_YAML} --output_dir ${PROJECT_DIR} --container_basename "launcher-${RANDOM_STRING}"

# launch docker
cd "${PROJECT_DIR}"
docker compose --project-name ${RANDOM_STRING} up -d

# show log
echo "Displaying log... press Ctrl+C to exit."
trap "docker compose --project-name ${RANDOM_STRING} down" SIGINT
docker compose --project-name ${RANDOM_STRING} logs -f
