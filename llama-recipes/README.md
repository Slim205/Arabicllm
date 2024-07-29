


## Fine tuning using FSDP :


torchrun --nnodes 1 --nproc_per_node 4 -m llama_recipes.finetuning --enable_fsdp --dataset "custom_dataset" --custom_dataset.file "/gpfs/users/barkallasl/ift/llama-recipes/aya_dataset.py" --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --dist_checkpoint_root_folder "/gpfs/workdir/barkallasl/model_checkpoints" --dist_checkpoint_folder fine-tuned --fsdp_config.pure_bf16 --use_fast_kernels --use_wandb


## TO HF : 
 python -m llama_recipes.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path  /gpfs/workdir/barkallasl/model_checkpoints_cidar/fine-tuned-meta-llama/Meta-Llama-3-8B-Instruct --consolidated_model_path /gpfs/workdir/barkallasl/Evaluation_ift_cidar/Meta-Llama-3-8B-Instruct --HF_model_path_or_name Slim205/meta-llama-Aya

## Evaluation : 
python eval.py --model vllm --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.5,max_model_len=4096" --limit 100 --open_llm_leaderboard_tasks --output_path results --log_samples --batch_size auto  

accelerate launch --multi_gpu --num_processes=4 run_evals_accelerate.py --model_args="pretrained=/gpfs/workdir/barkallasl/Evaluation_ift_cidar/Meta-Llama-3-8B-Instruct" --custom_tasks community_tasks/arabic_evals.py --tasks examples/tasks/arabic_tasks_0.txt --override_batch_size 8 --save_details --output_dir="./output_arabic_benchmark_new" --dataset_loading_processes 10
