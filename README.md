# In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations

This repository contains the code and data for the paper "In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations". The project investigates how Large Language Models (LLMs) exhibit latent preferences for different information sources, particularly in news article selection, academic paper ranking and ecommerce product recommendation tasks.

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LLM-Latent-Source-Preferences
```

2. Create a `.env` file with your API keys (if you wish to test OpenAI models):
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

If you wish to use the Azure OpenAI service you will need to add the following - 

```bash
AZURE_ENDPOINT_URL=your_azure_endpoint_url_here
AZURE_OPENAI_SUBSCRIPTION_KEY=your_azure_subscription_key_here
```

3. To run experiments using the OpenAI APIs you can just install the dependencies mentioned in `requirements.txt`. The experiments were run using Python 3.11.2

For experiments involving local models we use the SGLang docker container -

```
docker run --gpus all -it \
    --shm-size 32g \
    -v REAL_PATH:PATH_INSIDE_CONTAINER \
    --env "HF_TOKEN=YOUR_HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    bash
```

At the time of running these experiments, latest pointed to `v0.5.0rc2-cu126`

---

## Indirect Experiments

Folder: `Indirect_Experiments/`

### How to run?

Simply run the bash file corresponding to the dataset you wish to run for. You can choose between different LLMs by commenting out the existing one/adding new ones.

Note: If you wish to use LLMs via Azure please add the API Keys etc as above and in the model list in `runner_X.sh` bash file prepend the model name with `azure--`. This will use the azure deployment instead of a local one for that model. The save folder name however will remove this `azure--` prefix and only use the rest of the model name as the folder name.

---

## Direct Experiments

Folder: `Direct_Experiments/`

### How to run?

Same as Indirect experiments

---

## Case Study 1: AllSides News Choice

Folder: `Case_Study_1_AllSides_News_Choice/`

### How to run?

Simply run the bash file. You can choose between different LLMs by commenting out the existing one/adding new ones as well as which experiments to run by commenting out the ones you do not wish to run in the experiment_types list.

---

## Case Study 2: Amazon Seller Choice

Folder: `Case_Study_2_Amazon_Seller_Choice/`

### How to run?

Simply run the bash file. You can choose between different LLMs by commenting out the existing one/adding new ones as well as which experiments to run by commenting out the ones you do not wish to run in the experiment_types list.

---

## Data and Artifacts

Folder: `Artifacts/`

Contains all the datasets used by different experiments.

---

## Outputs

Folder: `Outputs/`

The scripts will save their outputs in this folder.

---

## Results

We provide all experimental results in compressed form. During experimentation, each LLM inference output was saved as an individual JSON file to simplify debugging and reruns. However, distributing millions of JSON files is impractical, so we aggregate them into grouped files for sharing.

Due to the large storage requirements, all results are hosted on HuggingFace:
*[aflah/LLM-Latent-Preferences](https://huggingface.co/datasets/aflah/LLM-Latent-Preferences)*

### Structure

* **`A/`** – Results from the **Direct** experiments.
  Each subfolder corresponds to one of the four tasks. Files within these subfolders follow the naming pattern:
  `MODEL_NAME=BADGE_NAME=SEED`
  Files are stored in Arrow IPC (Feather v2) format using Polars’ `write_ipc` API. Use Polars’ `read_ipc` API to load them.

* **`B/`** – Results from the **Indirect** experiments.
  The directory layout mirrors that of `A/`.

* **`Case_Study_1_All_Sides`** – Results for **Case Study 1: AllSides News Choice**.
  Each file is a JSON named using the convention:
  `MODEL_CONFIG_SEED.json`

* **`Case_Study_2_Amazon`** – Results for **Case Study 2: Amazon Seller Choice**.
  The file structure and format match those in the `A/` directory.

---

## Repository structure

```
Artifacts/                      # Standardized data and metadata used by experiments
Indirect_Experiments/           # Indirect preference probes (latent signals)
Direct_Experiments/             # Direct preference elicitation (explicit prompts)
Case_Study_1_AllSides_News_Choice/   # Case Study with AllSides Data
Case_Study_2_Amazon_Seller_Choice/   # Case Study with Amazon Seller Data
Outputs/                        # Results for Experiments are Saved Here
README.md
```

The code for an earlier version accepted to `ICML 2025 Workshop on Reliable and Responsible Foundation Models` is present under the `ICML-R2-FM` branch of the repository.

---

## 📝 Citation

If you would like to cite our work, please use the following BibTeX entry:

```bibtex
@inproceedings{khan2025agents,
  title={In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations},
  author={Khan, Mohammad Aflah and Amani, Mahsa and Das, Soumi and Ghosh, Bishwamittra and Wu, Qinyuan and Gummadi, Krishna P and Gupta, Manish and Ravichander, Abhilasha},
  booktitle={ICML 2025 Workshop on Reliable and Responsible Foundation Models}
}
```
