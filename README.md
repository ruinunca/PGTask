# PGTask: Introducing the Task of Profile Generation from Dialogues

In this project, we build a Profile Generation model for the Profile Generation Task (PGTask).
We train a model in the [Profile Generation Dataset (PGD)](https://tinyurl.com/PGDataset) dataset, which is based in the `Persona-Chat` dataset.

**PGTask: Introducing the Task of Profile Generation from Dialogues**. [[paper]](https://arxiv.org/abs/2304.06634)

## Citation
If you find PGTask useful in your work, please cite the following paper:
```
@article{ribeiro2023pgtask,
      title={PGTask: Introducing the Task of Profile Generation from Dialogues}, 
      author={Rui Ribeiro and Joao P. Carvalho and Luísa Coheur},
      journal={arXiv preprint arXiv:2304.06634},
      year={2023},
}
```

## Requirements:

This project uses Python 3.9+

Create a virtual environment with:

```bash
python3 -m virtualenv venv
source venv/bin/activate
```

Install the requirements (inside the project folder):
```bash
git clone git@github.com:ruinunca/PGTask.git
cd PGTask
pip install -r requirements.txt
```

## Getting Started:

### Download PGDataset

First, download PGDataset dataset from [this link](https://tinyurl.com/PGDataset) or use the command bellow.
Create a folder named `data/` on the project root and place it there:

```bash
mkdir data/
cd data/
wget https://web.tecnico.ulisboa.pt/rui.m.ribeiro/data/PGDataset.zip
unzip PGDataset.zip
```

## Train the model:
**Note:** 
Due to the `EarlyStopping`, you need to run the script with both `--do_train` and `--do_eval`.

```bash
(venv) python run_profile_generation.py \
--do_train \
--do_eval \
--model_name_or_path microsoft/DialoGPT-small \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
```

Available commands:

Training arguments:
```bash
optional arguments:
  --seed                          Training seed.
  --per_device_train_batch_size   Batch size to be used at training.
  --per_device_eval_batch_size    Batch size to be used at training.
  --model_name_or_path            Model to use
```

See more arguments at `run_profile_generation.py` and [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) from HuggingFace.

This will create a folder containing the model fine-tuned on the PGDataset dataset at `experiments/profile_generation`.

## Evaluation:

For evaluation of the model, first we generate the results, and then we use a different script to get the evaluation scores.
The reason for this approach is that if we want to experiment different metrics, we don't have to generate each time.

Generate the results:
```bash
python generate_results.py \
--model_type gpt2 \
--model_name_or_path experiments/experiment_%Y-%m-%d_%H-%M-%S \
```

This command generates a `generated_results.json` file inside the experiment's folder with the examples from the test dataset.

Evaluate the results:
```bash
python evaluate_results.py \
--experiment experiments/experiment_%Y-%m-%d_%H-%M-%S \
```

This command generates a `generated_scores.csv` and `generated_scores.json` files inside the experiment's folder with the results from the test dataset.

### Interact

You can also interact with the bot and experiment the profile generation using `interact.py`.


## Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/"
```

## References
- [BEauRTy](https://github.com/ruinunca/BEauRTy), a simple classification repository using BERT.
- [Hugging Face](https://huggingface.co/docs/transformers/model_doc/bert).
