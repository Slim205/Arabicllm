# Arabicllm

To run the Arabic Leaderboard benchmark, please use the following commands:

1. Set the configuration to Multiple GPUs using the following command:

    ```bash
    accelerate config
    ```

2. Run the following command:

    ```bash
    accelerate launch --multi_gpu --num_processes=2 run_evals_accelerate.py --model_args="pretrained=EleutherAI/pythia-70m,model_parallel=True" --custom_tasks community_tasks/arabic_evals.py --tasks examples/tasks/arabic_tasks_0.txt --override_batch_size 32 --save_details --output_dir="./output_arabic_benchmark" --dataset_loading_processes 10
    ```

This command creates a `.json` file in the `output_arabic_benchmark` directory containing the results of the evaluations of the model `EleutherAI/pythia-70m` over all 137 tasks (in `arabic_tasks_0.txt`).
