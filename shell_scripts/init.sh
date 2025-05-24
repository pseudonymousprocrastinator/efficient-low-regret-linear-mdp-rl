#!/bin/bash
if [[ "$#" -gt 0 ]]; then
  cd "$1"
fi
CODE_DIR="$(pwd -P)"
echo "Code root directory: $CODE_DIR"
echo 'Downloading linearized model files...'
wget -O lsrl_data.zip 'https://www.dropbox.com/scl/fo/4jzi9rwa4az380vx7jl99/ALp6GHjCz0TXnHUJ_lS0dm4?rlkey=3pskn9acubbqjyawi4d4gzt1l&st=bs8lwz4m&dl=1'
unzip lsrl_data.zip -x / -d lsrl_data
mv -f lsrl_data/linearized_*.dat models/
rm -f lsrl_data/linearized_*.dat
rmdir lsrl_data/
rm lsrl_data.zip
echo "Model files downloaded and copied to $CODE_DIR/models/ dir..."
