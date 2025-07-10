# In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations

This repository contains the code and data for the paper "In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations". The project investigates how Large Language Models (LLMs) exhibit latent preferences for different information sources, particularly in news article selection and academic paper ranking tasks.

## 🎯 Overview

This research explores whether LLMs have implicit biases toward certain information sources when making decisions. Through controlled experiments, we examine how models choose between articles from news sources with different political leanings (left, center, right) and how they rank academic papers from venues with varying prestige metrics.

## 📂 Repository Structure

```
LLM-Latent-Source-Preferences/
├── base_run.py                    # Main experiment for news article preference analysis
├── rank_news.py                   # News ranking experiments with source badges
├── rank_venues.py                 # Academic venue ranking experiments
├── rank_venues_with_context.py    # Academic ranking with contextual information
├── Dataset/                       # Experimental data and mappings
│   ├── standardized_dsde_*.json          # Academic paper datasets
│   ├── top_20_sources_per_leaning_*.json # News source classifications
│   ├── top_10_conferences_*.json         # Academic venue rankings
│   └── ...                              # Additional metadata files
└── README.md
```

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LLM-Latent-Source-Preferences
```

2. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

3. For non-GPT models, set up SGLang using Docker:

```bash
docker run --gpus all -it \
    --shm-size 32g \
    -v REAL_PATH:PATH_INSIDE_CONTAINER \
    --env "HF_TOKEN=YOUR_HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    bash
```

Replace `YOUR_HF_TOKEN` with your Hugging Face token for accessing gated models.

## 🚀 Usage

### News Article Preference Experiments

Run the base experiment to analyze LLM preferences for news articles:

```bash
# Test mode with GPT-4
python base_run.py --model gpt-4o-mini-2024-07-18 --seed 42 --mode test

# Production mode with a Hugging Face model
python base_run.py --model meta-llama/Llama-2-7b-chat-hf --seed 42 --mode prod
```

### News Source Ranking with Badges

Analyze how source metadata (badges) influences LLM preferences:

```bash
python rank_news.py --data_domain politics --seed 42 --mode test --badge_to_use Base --model_name gpt-4o-mini-2024-07-18
```

Available badge types:
- `Base`: No additional information
- `X_Handle`, `X_Followers`, `X_URL`: Twitter/X metadata
- `Instagram_Handle`, `Instagram_Followers`, `Instagram_URL`: Instagram metadata
- `URL`: Website information
- `Year_of_Establishment`, `Years_Since_Establishment`: Temporal information

### Academic Venue Ranking

Evaluate LLM preferences for academic papers from different venues:

```bash
python rank_venues.py --seed 42 --mode test --badge_to_use Base --model_name gpt-4o-mini-2024-07-18
```

Available badge types for venues:
- `Base`: No additional information
- `H5-Index`: Google Scholar H5-Index rankings
- `H5-Median`: Google Scholar H5-Median rankings

### Academic Ranking with Context

Run experiments with additional contextual information:

```bash
python rank_venues_with_context.py --data_domain computational_linguistics --seed 42 --mode test --badge_to_use Base --model_name gpt-4o-mini-2024-07-18
```


## 📝 Citation

If you would like to cite our work, please use the following BibTeX entry:

```bibtex
@inproceedings{khan2025agents,
  title={In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations},
  author={Khan, Mohammad Aflah and Amani, Mahsa and Das, Soumi and Ghosh, Bishwamittra and Wu, Qinyuan and Gummadi, Krishna P and Gupta, Manish and Ravichander, Abhilasha},
  booktitle={ICML 2025 Workshop on Reliable and Responsible Foundation Models}
}
```