# In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations

This repository contains the code and data for the paper "In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations". The project investigates how Large Language Models (LLMs) exhibit latent preferences for different information sources, particularly in news article selection, academic paper ranking and ecommerce product recommendation tasks.

The paper was accepted at ICLR 2026 and IASEAI 2026 (Non-Archival). An earlier version of this paper was presented at ICML 2025 Workshop on Reliable and Responsible Foundation Models and the code accompanying the same is present under the `ICML-R2-FM` branch of the repository. 

<img width="2500" height="1224" alt="ICLR 2026 Poster - Agent Trust pptx" src="https://github.com/user-attachments/assets/32bfee2e-563e-446a-9796-93c5492eda3b" />


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

---

## 📝 Citation

If you would like to cite our work, please use the following BibTeX entry:

```bibtex
@inproceedings{
    khan2026in,
    title={In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer {LLM} Generations},
    author={Mohammad Aflah Khan and Mahsa Amani and Soumi Das and Bishwamittra Ghosh and Qinyuan Wu and Krishna P. Gummadi and Manish Gupta and Abhilasha Ravichander},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=yTUNl6jYGU}
}
```

If you use the Amazon Seller Choice Dataset then please cite the original paper for that dataset:

```bibtex
@article{10.1145/3686994,
author = {Dash, Abhisek and Chakraborty, Abhijnan and Ghosh, Saptarshi and Mukherjee, Animesh and Gummadi, Krishna P.},
title = {Investigating Nudges toward Related Sellers on E-commerce Marketplaces: A Case Study on Amazon},
year = {2024},
issue_date = {November 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {8},
number = {CSCW2},
url = {https://doi.org/10.1145/3686994},
doi = {10.1145/3686994},
abstract = {E-commerce marketplaces provide business opportunities to millions of sellers worldwide. Some of these sellers have special relationships with the marketplace by virtue of using their subsidiary services (e.g., fulfillment and/or shipping services provided by the marketplace) -- we refer to such sellers collectively as Related Sellers. When multiple sellers offer to sell the same product, the marketplace helps a customer in selecting an offer (by a seller) through (a) a default offer selection algorithm, (b) showing features about each of the offers and the corresponding sellers (price, seller performance metrics, seller's number of ratings etc.), and (c) finally evaluating the sellers along these features. In this paper, we perform an end-to-end investigation into how the above apparatus can nudge customers toward the Related Sellers on Amazon's four different marketplaces in India, USA, Germany and France. We find that given explicit choices, customers' preferred offers and algorithmically selected offers can be significantly different. We highlight that Amazon is adopting different performance metric evaluation policies for different sellers, potentially benefiting Related Sellers. For instance, such policies result in notable discrepancy between the actual performance metric and the presented performance metric of Related Sellers. We further observe that among the seller-centric features visible to customers, sellers' number of ratings influences their decisions the most, yet it may not reflect the true quality of service by the seller, rather reflecting the scale at which the seller operates, thereby implicitly steering customers toward larger Related Sellers. Moreover, when customers are shown the rectified metrics for the different sellers, their preference toward Related Sellers is almost halved. We believe our findings will inform and encourage further deliberation toward more effective governance of such design choices and policies adopted by e-commerce marketplaces.},
journal = {Proc. ACM Hum.-Comput. Interact.},
month = nov,
articleno = {455},
numpages = {31},
keywords = {algorithmic auditing, choice architecture, e-commerce marketplace, nudges, preferential treatment, related sellers}
}
```
