#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

# Set this flag to True to only evaluate the model and not train it
EVAL_ONLY=False
RECOMPUTE_METRICS=True
ALL_TOPICS=(50)

eval_flag=""
if [ "$EVAL_ONLY" = "True" ]; then
    eval_flag="--eval_only"
fi

recompute_flag=""
if [ "$RECOMPUTE_METRICS" = "True" ]; then
    recompute_flag="--recompute_metrics"
fi

for NUM_TOPICS in ${ALL_TOPICS[@]}; do
    # python run_topic_model_baselines.py \
    #     --model lda \
    #     --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
    #     --results_path results/dbpedia_14/lda_K${NUM_TOPICS} \
    #     --num_topics ${NUM_TOPICS} \
    #     --num_seeds 5 \
    #     $eval_flag \
    #     $recompute_flag

    python run_topic_model_baselines.py \
        --model prodlda \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/prodlda_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        $eval_flag \
        $recompute_flag

    python run_topic_model_baselines.py \
        --model zeroshot \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/zeroshot_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        $eval_flag \
        $recompute_flag

    python run_topic_model_baselines.py \
        --model combined \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/combined_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        $eval_flag \
        $recompute_flag

    # python run_topic_model_baselines.py \
    #     --model etm \
    #     --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
    #     --results_path results/dbpedia_14/etm_K${NUM_TOPICS} \
    #     --num_topics ${NUM_TOPICS} \
    #     --num_hidden_layers 2 \
    #     --num_seeds 5 \
    #     $eval_flag \
    #     $recompute_flag

    python run_topic_model_baselines.py \
        --model bertopic \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/bertopic_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_seeds 5 \
        $eval_flag \
        $recompute_flag


    python run_topic_model_ours.py \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/Llama-3.2-3B-Instruct/${NUM_TOPICS}_CE \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type CE \
        --temperature 3 \
        --loss_weight 1000 \
        $eval_flag \
        $recompute_flag

    python run_topic_model_ours.py \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/Llama-3.2-3B-Instruct/${NUM_TOPICS}_KL \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type KL \
        --temperature 3 \
        --loss_weight 1000 \
        $eval_flag \
        $recompute_flag

    python run_topic_model_ours.py \
        --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/Llama-3.2-1B-Instruct/${NUM_TOPICS}_CE \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type CE \
        --temperature 3 \
        --loss_weight 1000 \
        $eval_flag \
        $recompute_flag

    python run_topic_model_ours.py \
        --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/Llama-3.2-1B-Instruct/${NUM_TOPICS}_KL \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type KL \
        --temperature 3 \
        --loss_weight 1000 \
        $eval_flag \
        $recompute_flag
done
