# ArabicLLM

## Project Overview
The goal of the project was to adapt large language models for the Arabic language and create a new state-of-the-art Arabic LLM. Due to the scarcity of Arabic instruction fine-tuning data, not many LLMs have been trained specifically in Arabic, which is surprising given the large number of Arabic speakers.  
Our final model was trained on a high-quality instruction fine-tuning (IFT) dataset, generated synthetically and then evaluated using the Hugging Face Arabic leaderboard.

## Details
The methodology introduced:  
- Generate synthetic data in English  
- Translate the data to Arabic using NLLB  
- Fine-tune the Gemma-2-2b-it model using the new data  
- Evaluate using the Arabic leaderboard benchmarks.

### Synthetic Data
This is the approach we used to construct our dataset:  
- Scraping Wikipedia articles about Arabic culture, science, Arab countries, religion, etc.  
- Generating questions and answers from paragraphs, with 75% of the paragraphs shortened, while ensuring the responses are comprehensive  
- Generating 3-turn conversational data using the remaining 25% of longer paragraphs: the first multi-turn dataset in Arabic (Slim205/multi_turn_nllb).

### Training Settings
The 2B version was trained for 2 days on 1 A100 GPU using LoRA with a rank of 128, a learning rate of 1e-4, and a cosine learning rate schedule.
Here is the information in table format:

## Datasets Used
The following datasets were used for instruction fine-tuning. Each dataset includes the number of instruction and answer tokens:

| **Dataset**                                  | **Instruction** | **Answer**  |
|----------------------------------------------|-----------------|-------------|
| Slim205/boolq_ift                            | 1,290,801       | 275,243     |
| Slim205/race_ift_v02_filtered                | 33,798,422      | 4,466,553   |
| Slim205/copa_ift_v02_filtered                | 11,663          | 11,703      |
| Slim205/hellaswag_ift_v02_filtered           | 7,264,646       | 1,448,588   |
| Slim205/sciq_ift_v02_filtered                | 1,559,005       | 276,289     |
| Slim205/arc_challenge_ift_v02_filtered       | 64,031          | 41,144      |
| Slim205/arc_easy_ift_v02_filtered            | 112,489         | 80,226      |
| Slim205/piqa_ift_v02_filtered                | 931,060         | 503,598     |
| Slim205/gsm8k_ift_v02                        | 435,759         | 563,772     |
| Slim205/wiki_data_full_filtered              | 1,934,101       | 10,900,116  |
| Slim205/multi_turn_v02_filtered              | 2,740,059       | 14,347,695  |
| **Total**                                    | **50,142,036**  | **32,914,927** |

## Model: BARKA-9B-it

The table below compares the performance of the base model **Gemma-2-9b-it** and a fine-tuned version across multiple benchmarks:

| **Benchmark**     | **Gemma-2-9b-it** | **Fine-tuned** |
|-------------------|-------------------|----------------|
| ACVA              | 48.98             | **73.55**      |
| AlGhafa           | 54.17             | **54.63**      |
| MMLU              | **53.33**         | 52.51          |
| EXAMS             | 50.65             | **52.89**      |
| ARC Challenge     | 55.00             | **58.88**      |
| ARC Easy          | 52.45             | **59.46**      |
| BOOLQ             | 83.99             | **86.38**      |
| COPA              | 60.00             | **62.22**      |
| HELLAWSWAG        | 32.33             | **38.00**      |
| OPENBOOK QA       | 50.30             | **56.16**      |
| PIQA              | 70.54             | **71.96**      |
| RACE              | 45.55             | **48.73**      |
| SCIQ              | **51.06**         | 50.35          |
| TOXIGEN           | 82.35             | **85.45**      |
| **All**           | 56.48             | **60.80**      |

## Evaluation

Evaluation script: using https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard benchmarks

    ```bash
    accelerate launch --multi_gpu --num_processes=2 run_evals_accelerate.py --model_args="pretrained=EleutherAI/pythia-70m,dtype=bfloat16" --custom_tasks community_tasks/arabic_evals.py --tasks examples/tasks/arabic_tasks_0.txt --override_batch_size 32 --save_details --output_dir="./output_arabic_benchmark" --dataset_loading_processes 10
    ```

This command will generate a `.json` file in the `output_arabic_benchmark` directory with the evaluation results for the model **EleutherAI/pythia-70m** over all 137 tasks listed in `arabic_tasks_0.txt`.

## Data Link

The full dataset for this project can be found at [Slim205/total_data_baraka_ift](#).

The 9B model  : https://huggingface.co/Slim205/BARKA-9b-it

The 2B model  : https://huggingface.co/Slim205/BARKA-2b-it 
