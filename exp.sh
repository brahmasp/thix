#!/bin/sh

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# launch code
echo "Running main.py"
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 main.py "$@"

# cp zip_dirs.sh wandb_tests/wandb/
# cd wandb_tests/wandb/
# ./zip_dirs.sh
# mv *.zip ../../
