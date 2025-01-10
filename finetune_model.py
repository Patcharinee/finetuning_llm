import os
import logging
import transformers
import torch
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import datetime

logger = logging.getLogger(__name__) ### still have some problem with logger ?

# Load environment variables
load_dotenv()

# Get huggingface token from environment variable
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_HUB_TOKEN")

### Set up the model, training config, tokenizer, split data into train dataset and test dataset ###
model_name = "EleutherAI/gpt-neo-125m" 
#"EleutherAI/pythia-70m"

tokenizer = AutoTokenizer.from_pretrained(
   model_name)

# Load question-answer pair dataset we have prepared from .csv file. 
dataset_filename = "training_data.csv"
dataset_for_finetune = load_dataset('csv', data_files=dataset_filename)

# Tokenize the dataset
def tokenize_function(examples):    
  if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
  elif "input" in examples and "output" in examples:
    text = examples["input"][0] + examples["output"][0]
  else:
    text = examples["text"][0]

  tokenizer.pad_token = tokenizer.eos_token
  tokenized_inputs = tokenizer(
      text,
      return_tensors="np",
      padding=True,
  )
  return tokenized_inputs

tokenized_dataset = dataset_for_finetune.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)
print("-----------tokenized_dataset")
print(tokenized_dataset)

# split data to train and test
tokenized_dataset["train"] = tokenized_dataset["train"].add_column("labels", tokenized_dataset["train"]["input_ids"])
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=123)
print("-----------splitted dataset")
print(split_dataset)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]
print("-----------Train dataset")
print(train_dataset)
print("-----------Test dataset")
print(test_dataset)



use_hf = False
dataset_path = dataset_filename

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

# Load the base model (the pretrained model one (not finetuned)) ###
print(f"loading base model: {model_name}")
base_model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"Base model-----")
print(base_model)

# count #of GPU that we have and use GPU if there is any (o.w. use CPU)
device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

# put the model on the device (GPU or CPU)
base_model.to(device) 

# Define function to carry out inference #

max_in_tokens = 1000    # max tokens to be input to the model
max_out_tokens = 100    # max tokens to be generated from the model

def inference(text, model, tokenizer, max_input_tokens=max_in_tokens, max_output_tokens=max_out_tokens):
  # Tokenize text input (quesion)
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device), # put the tokens of dataset to the same device as the model
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

### Setup training/finetuning ###
# max number of training steps == max number of batches of training data that we will run on the model
# step = batch of training data 
max_steps = len(train_dataset)
x = datetime.datetime.now()
time = f"{x.strftime("%Y%m%d")}_{x.strftime("%H%M%S")}"
trained_model_name = f"PyATS_{model_name}_{max_steps}_steps_{time}"  # name includes number of steps date time
output_dir = trained_model_name

training_args = TrainingArguments(
    # Learning rate
    learning_rate=1.0e-5,

    # Number of training epochs
    num_train_epochs=1,

    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    max_steps=max_steps,

    # Batch size for training
    per_device_train_batch_size=1,

    # Directory to save model checkpoints
    output_dir=output_dir,

    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=120, # Number of update steps between two evaluations
    save_steps=120, # After # steps model is saved
    warmup_steps=1, # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=1, # Batch size for evaluation
    eval_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps = 4,
    gradient_checkpointing=False,

    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False
    )

# Just to calculate the model FLOPS and memory footprint
model_flops = (
    base_model.floating_point_ops(
        {
            "input_ids": torch.zeros(
                (1, training_config["model"]["max_length"])
                )
        }
    )
  * training_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

# Trainer class to print out information during the model training
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

### FINETUNE THE MODEL ###
training_output = trainer.train()

### Save the trained model locally ###
save_dir = f'D:/dev/Finetuned_models/{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to:", save_dir)

# Load the trained model from local directory ###
finetuned_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True) 
# local_files_only = True means get the model from local directory not from Huggingface

# put the trained model on the device
finetuned_model.to(device) 

### Try to run the trained model on a test question to see whether it performs better ###
test_question = test_dataset[0]['question']
print("Question input (test):", test_question)

print("Finetuned model's answer: ")
print(inference(test_question, finetuned_model, tokenizer))

test_answer = test_dataset[0]['answer']
print("Target answer output (test):", test_answer)
