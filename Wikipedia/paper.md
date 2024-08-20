### Abstract : 
- Introduce instruction pretraining
- build an open-source model : synthesizer ==> construct 200M questions for 40+ topics
- analyse the instruction synthetizer and the instruction data to understand why they led to the success of their approach.

Vanilla pretraining : pretraining using only raw data
Intruction pretraining : pretrain on an augmented corpora : augment each raw text with a set of instruction-response pairs 

how they construct the synthetizer ? 
- converting datasets into this form : instruction + response + raw text 
- Finetuning a 7B model (Mistral-7Bv0.1)to construct instruction/response pairs based on a raw data
- use this synthetizer to create 200M instruction-response pairs accross +40 tasks

Pretraining from scratch : 
trained 500M model on 100B tokens ==> have the same performance of a 1B parameter pretrained on 300B tokens 

Domain adaptave continual pretraining : 
models pretrained with instruction pretraining gain more from instruction tuning : llama-3-8b  + instruction continual pretraining was better than  llama3-70b in finance + medicine. 

## Details of the synthetizer construction : 

### Data collection : 
- Considering datasets that perform a task using a context. 
- the contexts need to cover diverse domains such as encyclopedias, social media and academic tests, ect
- The tasks (instruction-response) should be diverse : commonsense reasoning - sentiment analysis - Math questions
- for each example in the dataset, they gather all the tasks that can correspond to the context 
- for each dataset, they sample 10K examples with the highest number of instruction-response pairs (avoid dataset predominance + enhance task diversity)
- Instruction-response pairs formats :  free form completion / mutiple choice / free form completition with CoT (chaine of thought) / Multiple choice with CoT

### Data format : 
- use the template <CON> {text} </CON> to wrap the raw text.
- design different template for different formats of instructions 
- \n\n is used to connect instruction-response pairs and link them with the raw text
- use <s> before the beginning of each example and </s> after the end of each example


### Tuning : 

- The instruction synthesizer was tuned using few-shot examples. 
- an example is a piece of raw data + instruction-response pairs. 
- each sequence fed to the synthetizer is concatenated with some examples from the same dataset


### Inference : 
multi round inference to create fewshot examples.
In each round we concatenate the examples generated in previous rounds. 


## Pretraining from scratch : 

convert only a part of the data to instruction-augmented due to the need for a large amount of data
- start from a corpos of 200M pieces of text with 100B tokens
- Constructing the data with two rounds : converting 20M  raw texts first round + converting another 20 M raw texts the second round using the generated examples from the previous round as examples. 

### result : 
- 200M synthesized paris
### Evaluation : 
- run the evaluation on 500M model with vanilla Pretraining / Instruct pretraining and mix PT.
- run an evaluation on a 1.3B model using vanilla pt and instruct pt
- The instruct pretrained model is better 

## Domain adaptive continual pretraining : 
convert all the raw corpora into instruction-augmented corpora

- 3 round inference + consider 1/3 raw data each round

- Evaluation : Instruct pretraining on llama 3-8b outperform llama -70 b for some metrics


## Analysis: 
### Synthetizer :
compute the f1 similarity of the generated response and the gold response. 
- with zero shot : using only raw data
- with few shots : few examples of a raw text + instruction response pairs. 
- They evaluated the performance on the seen and unseen data 

### Instruction-Augmented Corpora : 
- sample 500 instruction-augmented texts from the augmented corpora and use GPT-4  to evaluate the synthesized instruction-response pairs. (relevance + accuracy)

General tasks used : 
Commesence reasoning- Coreference resolution - Natural language inference - Struct-to-Text - Summarization - Sentiment analysis - maths - code 
