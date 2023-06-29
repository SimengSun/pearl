
PROCESS=$1
echo $PROCESS

# action mining
if [ $PROCESS = "action_mining" ]; then
echo "action mining"
OUTPUT_FILE=./output/mined_actions_init.txt
python pearl.py --stage mine_actions --output-file $OUTPUT_FILE
fi

# action simplification
if [ $PROCESS = "action_simplification" ]; then
echo "action simplification"
INPUT_FILE=./output/mined_actions_init.txt
OUTPUT_FILE=./output/mined_actions_simplified.txt
SHARD_SIZE=80
python pearl.py --stage simplify_actions \
                --input-file $INPUT_FILE \
                --output-file $OUTPUT_FILE \
                --shard-size $SHARD_SIZE
# if simplification for multiple rounds, use the output file as the input file and repeat
fi

# refine demonstration
if [ $PROCESS = "refine" ]; then
echo "refine demonstration"
INPUT_FILE=./output/mined_actions_simplified.txt
OUTPUT_FILE=./output/pearl_out
PLAN_PROMPT=./prompt_bank/plan_gen_self_refine.txt
INVAL_PLAN_PROMPT=./prompt_bank/plan_gen_self_refine_w_invalid.txt
python pearl.py --stage refine \
                --output-file $OUTPUT_FILE \
                --prompt-plan-file $PLAN_PROMPT \
                --prompt-plan-invalid-file $INVAL_PLAN_PROMPT
fi

if [ $PROCESS = "baseline_mcq" ]; then
echo "baseline mcq"
OUTPUT_PREFIX=./output/baseline_mcq_out
python pearl.py --stage baseline_mcq --output-file $OUTPUT_PREFIX
fi

if [ $PROCESS = "baseline_gqa" ]; then
echo "baseline gqa"
OUTPUT_PREFIX=./output/baseline_gqa_out
python pearl.py --stage baseline_gqa --output-file $OUTPUT_PREFIX
fi

# PEARL: plan generation and execution, and evaluation with mapped accuracy
if [ $PROCESS = "pearl" ]; then
echo "PEARL main process"
OUTPUT_PREFIX=./output/pearl_out
PLAN_PROMPT=./prompt_bank/plan_gen.txt
INVAL_PLAN_PROMPT=./prompt_bank/plan_gen_w_invalid.txt
python pearl.py --stage pearl --output-file $OUTPUT_PREFIX  \
                --prompt-plan-file $PLAN_PROMPT \
                --prompt-plan-invalid-file $INVAL_PLAN_PROMPT 
fi
