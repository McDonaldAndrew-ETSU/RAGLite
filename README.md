# RAGLite

> **A Lightweight, Modular RAG Framework Built with a Deep Reasoning 1.5B LM, QLoRA Coldstart Finetuning, and Custom Vector Search - Runs 30+ Tokens/S on Just 4GB VRAM**

![RAGLite Banner](https://img.shields.io/badge/DeepReasoning-1.5B-blue?style=flat-square)
![QLoRA](https://img.shields.io/badge/QLoRA-Finetuned-success?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-Enabled-orange?style=flat-square)
![RAM](https://img.shields.io/badge/4GB-VRAM-lightgrey?style=flat-square)
![Portable](https://img.shields.io/badge/Portable-Yes-9cf?style=flat-square)
![License](https://img.shields.io/github/license/McDonaldAndrew-ETSU/RAGLite?style=flat-square)

---

**RAGLite** is a plug-and-play Retrieval-Augmented Generation framework that combines:

- 💡 **Distilled Qwen 1.5B** base model to lightweight, serverless, high-performance LLM (No API's Needed!)
- 🧠 **QLoRA finetuning** on coldstart .jsonl data
- 🔎 **Custom vector store** with cosine similarity for fast retrieval
- ⚙️ **Modular pipeline** for preprocessing, embedding, finetuning, and generation
- ⚡️ **Lightning Fast** functionality at over 30+ tokens generated per second on **just 4GB of VRAM**
- 🪶 **Featherweight** framework with serverless architecture.

> Designed for researchers, students, and developers building local-first, efficient, and powerful AI applications.

## Setup Environment

1. Install Python 3.11.8
2. If you have multiple versions of Python installed, you can specify which one to run:
   Check installed versions:
   `py -0`
   Run a specific version:
   `py -3.10`
   To make a specific version default:
   Update the PATH environment variable to point to the desired version.

3. First create virtual venv to separate project dependencies from your global dependencies (keeps somewhat a level of organization)
   `python -m venv .venv`

4. Activate the .venv:
   if Windows machine: `.venv\Scripts\activate`
   (I am using a git bash integrated terminal on a Windows machine - a somewhat hybrid terminal of traditional Linux commands - but the file structure/pathing on Windows to activate/deactivate a .venv stays the same - my command is `source .venv/Scripts/activate`)
   if UNIX/Linux/MacOS: `source .venv/bin/activate`

5. Ensure your coding environment (I am using VS Code) is using your Python Interpretor as your created .venv
   For VS Code: Ctrl + P, then type: `> Python: Select Interpretor`
   Additionally if using VS Code make sure you have the official Python extensions: Python, Python Debugger, Pylance

## Important Dependencies

1. Ensure you have the NVIDIA GPU Computing Toolkit version 12.6 installed to use your local GPU

2. Specifically run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- This is the most important package and doesn't install correctly when using a requirements.txt file

3. Install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

4. You should not receive any errors trying to install the requirements.txt asking for different python versions for specific dependencies.
   However if you do, I am sorry, you may have to manually handle installing the specific dependencies. Contact me if facing any issues.

## Downloading DeepSeek models from HuggingFace

- You can use any llm from hugging face; this NLP project we have so far been using open-source DeepSeek llms and their Distilled versions of their models.
- Here is their HuggingFace repository containing all their models: https://huggingface.co/deepseek-ai
- The latest DeepSeek models are their `DeepSeek-R1` collection. This is where I use their 1.5B model
- You may store any regular model architecture within the empty `deepseek/models` directory.
  - This is where we can pull entire models stored locally, and then quantize and cache them into a separate directory.

### 1.5B DeepSeek Model

1. Go to https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B for the 1.5B model
2. You may clone the repository (separate or within this project folder) and use Git LFS to pull down the heavy .safetensor files. Or you may download each file manually.
3. Store the 1.5B model files, including its .safetensor files within `deepseek/models/DeepSeek-R1-Distill-Qwen-1.5B`, the name of thw downloaded model.

## Preprocessing Pipeline

1. I have built a Jupyer Notebook `preprocessing_pipeline.ipynb` to handle cleaning the student data .json files.
2. This pipeline only processes one file at a time manually after inputting its file name.
   - This can be expanded on later when the model tuning is finished.
3. To use the notebook, simply import a student json file within the same top-level directory the notebook is in.
4. Follow the cell prompts and input your filename.
5. The json should then finish processing, removing any NaN contamination and be ready for use.
6. The file should be automatically written to the top-level `data` directory with "cleaned\_" prefixing the file name.

## Next Steps

Congratulations! You may now enter the `deepseek` directory to learn the code base and how to use this code.
