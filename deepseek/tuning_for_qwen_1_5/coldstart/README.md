# `coldstart` directory

1. This directory is a finetuning directory that uses the "Coldstart" technique. Coldstart is where a few hundred datainstances that have an emphasis on quality are used for the training of a LM. The following files and directories with their descriptions are otulined below.

- `coldstart.ipynb` - A user friendly Jupyter notebook that allows the user to train a LM on a list of `.jsonl` files for finetuning. Each block is chunked intuitively to understand the coldstart technique process.
- `generate_coldstart.py` - A script to generate fake data that represents academic advisor queries. It's output is `coldstart_data`.
- `coldstart_data` directory - Directory that contains .jsonl coldstart data files
