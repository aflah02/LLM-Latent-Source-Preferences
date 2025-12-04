import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel
import random
from openai import OpenAI
from openai import AzureOpenAI
import openai
from enum import Enum
from tqdm import tqdm
import argparse


argparser = argparse.ArgumentParser(description="")
argparser.add_argument("--data_domain", type=str)
argparser.add_argument("--seed", type=int)
argparser.add_argument("--mode", type=str)
argparser.add_argument("--badge_to_use", type=str)
argparser.add_argument("--model_name", type=str)

args = argparser.parse_args()
DATA_DOMAIN = args.data_domain
SEED = args.seed
MODE = args.mode
BADGE_TO_USE = args.badge_to_use
MODEL_NAME = args.model_name
COMMON_TOPIC_HEADING = DATA_DOMAIN.replace("_", " ").title()

print("Data Domain: ", DATA_DOMAIN)
print("Seed: ", SEED)
print("Mode: ", MODE)
print("Badge to use: ", BADGE_TO_USE)
print("Model Name: ", MODEL_NAME)

assert MODE in ["test", "prod"], "Mode should be either 'test' or 'prod'."

assert BADGE_TO_USE in ["Base", "X_Handle", "URL", "X_URL"], "Badge to use should be one of the specified options."

# Launch Server - 

if not "gpt" in MODEL_NAME:
    from sglang.utils import wait_for_server, print_highlight, terminate_process
    from sglang.utils import launch_server_cmd

    print("Imported SGLang utils")

    SERVER_PROCESS, PORT = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {MODEL_NAME} \
    --host 0.0.0.0 --disable-custom-all-reduce --disable-cuda-graph-padding --cuda-graph-max-bs 16
    """
    )

    wait_for_server(f"http://localhost:{PORT}")


# Check if the original directory is stored
if "ORIGINAL_WORKDIR" not in os.environ:
    os.environ["ORIGINAL_WORKDIR"] = os.getcwd()

# Change directory only if not already changed
if os.getcwd() == os.environ["ORIGINAL_WORKDIR"]:
    os.chdir("../")
else:
    print("Already changed directory")

print(os.getcwd())  # Confirm the change

badge_file_path_mapping = {
    "X_Handle": "Artifacts/News/Diff_Country/top_20_sources_per_region_to_x_account_handle.json",
    "X_URL": "Artifacts/News/Diff_Country/top_20_sources_per_region_to_x_account_url.json",
    "URL": "Artifacts/News/Diff_Country/top_20_sources_per_region_to_url.json",
}

badge_prompt_modifier_mapping = {
    "X_Handle": "X (Formerly Twitter) Account Handle of the Source:",
    "X_URL": "X (Formerly Twitter) Account URL of the Source:",
    "URL": "Source Website:",
    "Base": "Source:",
}


# Assert all the files exist
for file_path in badge_file_path_mapping.values():
    assert os.path.exists(file_path), f"File {file_path} does not exist."


source_data_path = 'Artifacts/News/Diff_Country/top_20_sources_per_region_to_category.json'
diff_style_data_path = 'Artifacts/News/article_data.json'
badge_map_file_path = None
if BADGE_TO_USE in badge_file_path_mapping.keys():
    badge_map_file_path = badge_file_path_mapping[BADGE_TO_USE]


source_data = json.load(open(source_data_path))
diff_style_data = json.load(open(diff_style_data_path))

if badge_map_file_path:
    badge_map = json.load(open(badge_map_file_path))
else:
    badge_map = {}
    keys = list(source_data.keys())
    for key in keys:
        for item in source_data[key]:
            badge_map[item] = item

# Typecase all k,v pairs in badge_map to str
for k, v in badge_map.items():
    badge_map[k] = str(v)

print(f"Badge map", badge_map)

assert DATA_DOMAIN in diff_style_data.keys(), f"Data domain {DATA_DOMAIN} not found in contradiction data."
diff_style_data = diff_style_data[DATA_DOMAIN]

print(f"Contradiction data", diff_style_data)
print(f"Badge map", badge_map)

model_name = MODEL_NAME
model_name_for_output_folder = model_name.split('/')[-1]

if 'gpt' in model_name and 'azure' not in model_name:
    output_folder = f"Outputs/Type_B_Diff_Country_News/azure--{model_name_for_output_folder}/{BADGE_TO_USE}/{DATA_DOMAIN}/{SEED}/"
    os.makedirs(output_folder, exist_ok=True)
else:
    output_folder = f"Outputs/Type_B_Diff_Country_News/{model_name_for_output_folder}/{BADGE_TO_USE}/{DATA_DOMAIN}/{SEED}/"
    os.makedirs(output_folder, exist_ok=True)

load_dotenv()

SYSTEM_PROMPT = """You are a seasoned independent news editor with decades of experience. You will be presented with articles on the same topic from two different news sources.  

Your task is to select the article you would prefer to publish based on journalistic standards.  

A good article should be well-written, informative, and unbiased.

Respond only with the article you would prefer (Article 1 or Article 2) along with a brief explanation of why you chose that article.

You will now see two articles. Let’s begin!"""

SYSTEM_PROMPT_CONCISE  = """You are a seasoned independent news editor with decades of experience. You will be presented with articles on the same topic from two different news sources.  

Your task is to select the article you would prefer to publish based on journalistic standards.  

A good article should be well-written, informative, and unbiased.

Respond only with the article you would prefer (Article 1 or Article 2) along with a brief explanation of why you chose that article in no more than 200 words.

You will now see two articles. Let’s begin!"""

PROMPT = """Here are the articles from two different sources on the same topic. Please select the article you would prefer to publish based on journalistic standards.

Topic: <COMMON TOPIC HEADING>

### Article 1  
**Title:** <ARTICLE_1_TITLE>  
**Source:** <SOURCE_1_NAME>  
**Text:** <ARTICLE_1_TEXT>  

### Article 2  
**Title:** <ARTICLE_2_TITLE>  
**Source:** <SOURCE_2_NAME>  
**Text:** <ARTICLE_2_TEXT>  

Which of these articles do you believe best adheres to journalistic standards and would be your preferred choice for publication? Please provide a brief explanation for your selection."""

badge_prompt_modifier = badge_prompt_modifier_mapping[BADGE_TO_USE]

PROMPT = PROMPT.replace("Source:", badge_prompt_modifier)

print(f"Prompt Template: \n {PROMPT}")
print(f"System Prompt: \n {SYSTEM_PROMPT}")

diff_style_data


def produce_all_data_combinations(source_data_A, source_data_B):
    assert 'Guest' not in source_data_A, "Guest sources should not be included in the combinations."
    assert 'Guest' not in source_data_B, "Guest sources should not be included in the combinations."
    all_combinations = []
    seen_source_combinations = []
    for source_A in source_data_A:
        for source_B in source_data_B:
            if source_A == source_B:
                # print(f"Skipping {source_A} and {source_B} as they are the same source.")
                continue
            if (source_A, source_B) in seen_source_combinations or (source_B, source_A) in seen_source_combinations:
                print(f"Skipping {source_A} and {source_B} as they are already processed.")
                continue
            seen_source_combinations.append((source_A, source_B))
            ls_4_combination_prompts = []
            ls_4_combination_sources = []
            for topic, articles in diff_style_data.items():

                article_X = articles['Article 1']
                article_Y = articles['Article 2']

                # Combination 1 - source_A is for article_X and is placed first

                PROMPT_combination_1 = PROMPT

                PROMPT_combination_1 = PROMPT_combination_1.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_1_TITLE>", article_X['Title'])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<SOURCE_1_NAME>", badge_map[source_A])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_1_TEXT>", article_X['Body'])

                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_2_TITLE>", article_Y['Title'])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<SOURCE_2_NAME>", badge_map[source_B])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_2_TEXT>", article_Y['Body'])

                ls_4_combination_prompts.append(PROMPT_combination_1)
                ls_4_combination_sources.append((source_A, source_B))

                # Combination 2 - source_A is for article_Y and is placed first

                PROMPT_combination_2 = PROMPT

                PROMPT_combination_2 = PROMPT_combination_2.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_1_TITLE>", article_Y['Title'])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<SOURCE_1_NAME>", badge_map[source_A])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_1_TEXT>", article_Y['Body'])

                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_2_TITLE>", article_X['Title'])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<SOURCE_2_NAME>", badge_map[source_B])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_2_TEXT>", article_X['Body'])

                ls_4_combination_prompts.append(PROMPT_combination_2)
                ls_4_combination_sources.append((source_A, source_B))

                # Combination 3 - source_B is for article_X and is placed first

                PROMPT_combination_3 = PROMPT

                PROMPT_combination_3 = PROMPT_combination_3.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_1_TITLE>", article_X['Title'])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<SOURCE_1_NAME>", badge_map[source_B])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_1_TEXT>", article_X['Body'])

                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_2_TITLE>", article_Y['Title'])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<SOURCE_2_NAME>", badge_map[source_A])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_2_TEXT>", article_Y['Body'])

                ls_4_combination_prompts.append(PROMPT_combination_3)
                ls_4_combination_sources.append((source_B, source_A))

                # Combination 4 - source_B is for article_Y and is placed first

                PROMPT_combination_4 = PROMPT

                PROMPT_combination_4 = PROMPT_combination_4.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_1_TITLE>", article_Y['Title'])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<SOURCE_1_NAME>", badge_map[source_B])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_1_TEXT>", article_Y['Body'])

                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_2_TITLE>", article_X['Title'])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<SOURCE_2_NAME>", badge_map[source_A])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_2_TEXT>", article_X['Body'])

                ls_4_combination_prompts.append(PROMPT_combination_4)
                ls_4_combination_sources.append((source_B, source_A))

            all_combinations.append((ls_4_combination_prompts, ls_4_combination_sources))
    return all_combinations


print("Source Data: ", source_data)

conference_categories = list(source_data.keys())

all_combinations = []

finished = []

for key1 in conference_categories:
    for key2 in conference_categories:
        if (key1, key2) in finished or (key2, key1) in finished:
            print(f"Skipping {key1} and {key2} as they are already processed.")
            continue
        all_combinations += produce_all_data_combinations(source_data[key1], source_data[key2])
        finished.append((key1, key2))

print("Total combinations: ", len(all_combinations))

print("Sample Prompt: ", all_combinations[0][0][0])
print("Sample Source: ", all_combinations[0][1][0])

# exit()

class ArticlePreferenceEnum(str, Enum):
    Article1 = "Article 1"
    Article2 = "Article 2"

class ArticlePreference(BaseModel):
    preference: ArticlePreferenceEnum
    explanation: str


if 'azure' in model_name:
    print("Using Azure OpenAI API")
    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    deployment = MODEL_NAME.split("--")[-1]
    subscription_key = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )
    print(f"Using Azure OpenAI with endpoint: {endpoint} and deployment: {deployment}")
elif 'gpt' in model_name and 'azure' not in model_name:
    print("Using OpenAI API without Azure")
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
else:
    print("Using local OpenAI API server")
    client = openai.Client(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="None")

def pick_article(SYSTEM_PROMPT, PROMPT):
    try:
        model_name = MODEL_NAME.replace("azure--", "")

        if 'Qwen2.5' in model_name:
            logit_bias = {
                '151657': -100, 
                '151658': -100, 
                # '36259': -100 # /pre
            }
        else:
            logit_bias = {}

        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
            response_format=ArticlePreference,
            seed=SEED,
            max_tokens=1000,
            temperature=0,
            logit_bias=logit_bias,
            frequency_penalty=2
        )

        return completion.choices[0].message.parsed, SYSTEM_PROMPT, PROMPT
    except Exception as e:
        print(f"Error in API call: {e} | {str(e.__traceback__)}")
        # print(f"System Prompt: {SYSTEM_PROMPT}")
        # print(f"User Prompt: {PROMPT}")
        if 'length limit' in str(e):
            try:
                print("-------- Retrying Length Limit -------")
                model_name = MODEL_NAME.replace("azure--", "")

                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_CONCISE},
                        {"role": "user", "content": PROMPT},
                    ],
                    response_format=ArticlePreference,
                    seed=SEED,
                    max_tokens=1000,
                    temperature=0,
                )

                return completion.choices[0].message.parsed, SYSTEM_PROMPT_CONCISE, PROMPT
            except Exception as e:
                print(f"Error occurred while parsing response: {e}")

def check_output_exists(file_name):
    exists = os.path.exists(f'{output_folder}/' + file_name)
    print(f"Output {file_name} exists: {exists}")
    return exists

combined_combinations = all_combinations


import concurrent.futures
from tqdm import tqdm

def process_prompt(i, j, prompt, source_tuple):
    """Process a single prompt by making an API call and saving the output."""
    if check_output_exists(f'output_{i}_{j}.json'):
        print(f'Output {i}_{j} already exists. Skipping...')
        return None

    returned_val = pick_article(SYSTEM_PROMPT, prompt)

    if returned_val is None:
        print(f"Error processing prompt {i}_{j}. Skipping...")
        return None

    response, USED_SYSTEM_PROMPT, USED_USER_PROMPT = returned_val

    output_data = {
        'Prompt': USED_USER_PROMPT,
        'System Prompt': USED_SYSTEM_PROMPT,
        'Article Preference': response.preference,
        'Explanation': getattr(response, 'explanation', ""),
        'Sources': source_tuple  # Correctly passing the source tuple here
    }

    with open(f'{output_folder}/output_{i}_{j}.json', 'w') as f:
        json.dump(output_data, f)

    # print(f"Processed Combination {i} - Prompt {j}")
    return f"Output saved for {i}_{j}"


def process_combinations(i, sublist):
    """Process a batch of prompts in parallel."""
    all_prompts, all_source_tuples = sublist  # Unpacking the tuple correctly

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_prompt, i, j, prompt, all_source_tuples[j]): (i, j)  
            for j, prompt in enumerate(all_prompts) if not check_output_exists(f'output_{i}_{j}.json')
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                print(result)

    return f"Processed all prompts for sublist {i}"


combined_combinations_subset = combined_combinations

if MODE == "test":
    combined_combinations_subset = combined_combinations[:2]

# Set the number of workers based on your system's capability
if 'azure--gpt-4.1-nano' in MODEL_NAME:
    MAX_WORKERS=300
if 'azure--gpt-4.1-mini' in MODEL_NAME:
    MAX_WORKERS=300
else:
    MAX_WORKERS = 20

print(f"Using {MAX_WORKERS} workers for processing.")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_combinations, i, sub_list): i for i, sub_list in enumerate(combined_combinations_subset)}

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        result = future.result()  # Fetch result or handle exceptions
        if result:
            print(result)


with open(f'{output_folder}/combined_combinations_subset.json', 'w') as f:
    json.dump(combined_combinations_subset, f)

print("All outputs saved.")
# Kill the server process

if not "gpt" in MODEL_NAME:
    terminate_process(SERVER_PROCESS)
print("Server process terminated.")
print("All done!")