# Setup and Familiarization

## Important Libraries

Run the following to ensure you have the correct torch backend:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes
pip install 'accelerate>=0.26.0'
```

These 3 modules (`torch` `torchvision` `torchaudio`) are commented out within `requirements.txt` due to install errors. Running these lines manually should install these dependencies without issue. Once complete, then run:

```bash
pip install -r requirements.txt
```

## General

1. Once you have finished downloading all files to for the model and put them in the `./models/DeepSeek-R1-Distill-Qwen-1.5B` directory, familiarize yourself with the various directories.

# `deepreason.py`

## Model choices

Starting at line 22 you will see

```python
llama_8b = "./models/DeepSeek-R1-Distill-Llama-8B"
qwen_7B = "./models/DeepSeek-R1-Distill-Qwen-7B"
qwen_1_5B = "./models/DeepSeek-R1-Distill-Qwen-1.5B"
cached = "./models/currently_cached_model"
trained = "./models/qwen_lora_trained"
```

These are path variables to help the user easily switch between the downloaded models within the `./models` directory.

## Comments

Please follow each method's documentation to find more about the method and what it does.

## How to use

1. Ensure your 1.5B model is in your `models` directory

2. For the following code at the end of the file:

```python
if __name__ == "__main__":
    llm = DeepReasonGenerator(model_name_or_path=trained, already_quantized=False)
    rag = RAGPipeline(vector_store_dir="./rag/simple_vector_store", generator=llm)

    while True:
        user_input = input("Rag Query? (Y/N): ")
        if user_input == "Y":
            query = input("Rag Query: ")
            response = rag.run(query)
        elif user_input == "N":
            prompt = input("You: ")
            response = llm.generate_response(prompt)
        elif user_input.lower() in ["exit", "quit"]:
            logger.info("Exiting chat. Have a good day!")
            break

```

- The class is preset with the `model_name_or_path` parameter with the `trained` variable. Feel free to adjust variables after downloading different models and saving them to the `./models` directory. You can see I have previously tested with the `llama_8B` and `qwen_7B` models.

- If the model is already downloaded, please keep the `already_quantized` parameter as `False`
  - Setting it to `True` allows the user to use an already Quantized version of the model that is saved in a `./currently_cached_model` directory.
  - This directory becomes populated after running, as the class is designed to immediately quantize the model from a model directory and then be used.
  - After the model is quantized and saved in `./currently_cached_model`, you may set the `already_quantized` parameter to `True` so it can pull the Quantized model a lot faster than having to quantize it every time.

3. To run this code, first cd into the `deepreason` directory.

4. Run:

```bash
python deep_seek.py
```

# `eval.ipynb` & `eval.py`

- `eval.py` helps with evaluation of the outputs designated in this file using BLEU and ROUGE.
- `eval.ipynb` shows the results between a base model output .jsonl file and a trained model output .jsonl file
  - Initially designed to show results of small amounts of coldstart on a model before training, and after training.

# `util` directory

1. Contains `prompts.py` - contains all methods to help prompt the llm with specific sections from the student data. So far, everything has been adjusted to where is brings back _somewhat consistent_ results. If it consistently brings back awful results for you, I suggest using a higher parameter model, though this current setup has been working for me, again emphasis on the _somewhat_.

2. Contains `stream_stop.py` - contains the class to return and format traditional ChatGPT like responses within your terminal window.

# `rag` directory

1. Use its README to find out more about the directory.

# `tuning` directory

1. Contains `coldstart` directory that contains necessary files to finetune a model on academic coldstart data found in the `data/coldstart_data` directory.

2. Contains `tuning.ipynb` the - contains my painstaking attempts at "taming" the unpredictable quantized 1.5B model. Play around with this if you like. I use a simple string comparison for evaluating performance, which isn't the best for hyperparameter tuning, but **plan to soon use methods like BLEU and ROUGE to make it more accurate**.

3. Contains logs, csv results, and findings notes based on the output of `tuning.ipynb`

# Next steps

Congratulations! This should be enough information to help you get started. If any questions please contact me.

- For **tuning the model**, I usually like to create a "tuning" directory for each model, hence the `tuning_for_qwen_1_5` directory.
- For **prompt engineering**, this can be done within `util/prompts.py`
