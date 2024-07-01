from vllm import LLM, SamplingParams
from multiprocessing import freeze_support, set_start_method
import ray,torch

num_gpus= torch.cuda.device_count()
print(num_gpus)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


#model_id = "EleutherAI/pythia-70m"
model_id ="meta-llama/Meta-Llama-3-70B-Instruct"

def try_llm(model_id) :

    llm = LLM(model=model_id,tensor_parallel_size=num_gpus,enforce_eager=True,distributed_executor_backend='ray')
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
   # freeze_support()
  #  set_start_method('spawn')
    ray.init(num_gpus=num_gpus)
    try_llm(model_id)
    ray.shutdown()

