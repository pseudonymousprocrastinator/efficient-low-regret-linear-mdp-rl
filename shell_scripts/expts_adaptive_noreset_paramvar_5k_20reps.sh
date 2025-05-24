#!/bin/bash

# Setting up the arguments
function print_usage { echo "Usage: $0 -m <model-filename> -o <output-prefix> [-n <num-jobs>=1(default) -r <num-reps>=1(default)]"; }

NUM_MP_JOBS=1
NUM_REPS=1
MODEL_FILENAME=""
OUTPUT_DIR="./output"
OUTPUT_PREFIX=""

[[ "$#" -eq 0 ]] && { print_usage; exit 1; }
while getopts ":hm:o:n:r:" opt; do
    case "${opt}" in
        m)
            MODEL_FILENAME="${OPTARG}"
            ;;
        o)
            OUTPUT_PREFIX="${OPTARG}"
            ;;
        n)
            NUM_MP_JOBS=${OPTARG}
            ;;
        r)
            NUM_REPS=${OPTARG}
            ;;
        h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unrecognised option"
            print_usage
            exit 1
            ;;
    esac
done
shift  $((OPTIND-1))

OUTPUT_DIR=${OUTPUT_DIR%/} # Strip trailing slash if necessary
OUTPUT_PREFIX=${OUTPUT_PREFIX#/} # Strip leading slash if necessary

if [[ -z "$MODEL_FILENAME" ]] || [[ -z "$OUTPUT_PREFIX" ]]; then
    echo "Model file name/output prefix not specified."
    print_usage
    exit 1
fi

function is_in_venv {
    local ISVENV=$(python -c 'import sys; print("y" if (hasattr(sys, "real_prefix") or sys.base_prefix != sys.prefix) else "n")' 2>/dev/null)
    [[ $ISVENV = "y" ]] && return 0 || return 1
}

function get_python_version {
    local PYVEROUT=$(python --version 2>/dev/null)
    local PYVERFMT="Python [0-9][0-9\.]*"
    [[ ! $PYVEROUT =~ $PYVERFMT ]] && return 1
    PYTHON_VER=$(echo $PYVEROUT | sed -r 's/Python ([0-9\.]+)/\1/')
    PYTHON_MAJ_VER=$(echo $PYVEROUT | sed -r 's/Python ([0-9]+\.[0-9]+)\..*/\1/')
    return 0
}

# Sanity checks
echo "Working directory: $(pwd -P)"
echo "Model filename: $MODEL_FILENAME"
echo "Num multiprocessing jobs: $NUM_MP_JOBS"
echo "Output directory: ${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-<algparams>"

# Check if running in python virtual env (not strictly necessary)
# is_in_venv
# [[ $? -eq 0 ]] || { echo 'Not in a python virtual environment.'; exit 1; }
# Check python version (should be 3.10.x)
# get_python_version
# [[ $? -eq 0 ]] || { echo 'Unable to get python version.'; exit 1; }
# [[ "$PYTHON_MAJ_VER" == "3.10" ]] || { echo "Python version should be 3.10.x (is currently $PYTHON_VER)."; exit 1; }
# Check for python code files
[[ -f experiment_base.py ]] || { echo "Unable to find experiment_base.py in working dir. Run this script from code root dir."; exit 1; }

# Running the actual experiments
mkdir -p "$OUTPUT_DIR"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-10-0p001-0p5"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 10 --alt-threshold 0.001 --learn-iters-budget-exp 0.5  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-50-0p001-0p5"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 50 --alt-threshold 0.001 --learn-iters-budget-exp 0.5 alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-10-1-0p5"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 10 --alt-threshold 1 --learn-iters-budget-exp 0.5  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-50-1-0p5"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 50 --alt-threshold 1 --learn-iters-budget-exp 0.5  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-10-0p001-0p6"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 10 --alt-threshold 0.001 --learn-iters-budget-exp 0.6 alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-50-0p001-0p6"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 50 --alt-threshold 0.001 --learn-iters-budget-exp 0.6  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-10-1-0p6"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 10 --alt-threshold 1 --learn-iters-budget-exp 0.6  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-50-1-0p6"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 50 --alt-threshold 1 --learn-iters-budget-exp 0.6  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-10-0p001-0p75"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 10 --alt-threshold 0.001 --learn-iters-budget-exp 0.75  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-50-0p001-0p75"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 50 --alt-threshold 0.001 --learn-iters-budget-exp 0.75  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-10-1-0p75"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 10 --alt-threshold 1 --learn-iters-budget-exp 0.75  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
python experiment_base.py -n $NUM_MP_JOBS -o "${OUTPUT_DIR}/${OUTPUT_PREFIX}-ad-nr-short-50-1-0p75"  --k-min 100 --num-reps $NUM_REPS --chunk-size 5 \
    --lookback-period 50 --alt-threshold 1 --learn-iters-budget-exp 0.75  alt_adaptive_noreset 5000 "$MODEL_FILENAME"
# END
