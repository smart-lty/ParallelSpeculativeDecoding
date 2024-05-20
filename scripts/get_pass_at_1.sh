dir=$1
mode=$2
dataset=$3
evaluate_functional_correctness exp/$dir/"$mode"_"$dataset".jsonl --problem_file=humaneval.jsonl > exp/$dir/results.log