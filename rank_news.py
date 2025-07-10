import os

# Run - pip install sglang-router

os.environ['HF_HOME'] = '/NS/llm-artifacts/nobackup/HF_HOME'



from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
import random
from openai import OpenAI
import openai
from enum import Enum
from tqdm import tqdm
import argparse
import os

argparser = argparse.ArgumentParser(description="CDE Standardized Experiment")
argparser.add_argument("--data_domain", type=str)
argparser.add_argument("--seed", type=int)
argparser.add_argument("--mode", type=str)
argparser.add_argument("--badge_to_use", type=str)
argparser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")

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

assert BADGE_TO_USE in ["Base", "X_Handle", "X_Followers", "X_URL","Instagram_Handle", "Instagram_Followers", "Instagram_URL", "URL", "Year_of_Establishment", "Years_Since_Establishment"], "Badge to use should be one of the specified options."

# Launch Server - 

if not "gpt" in MODEL_NAME:
    from sglang.utils import wait_for_server, print_highlight, terminate_process
    from sglang.utils import launch_server_cmd

    print("Imported SGLang utils")

    SERVER_PROCESS, PORT = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {MODEL_NAME} \
    --host 0.0.0.0
    """
    )

    # SERVER_PROCESS, PORT = launch_server_cmd(
    #     f"""
    # python3 -m sglang_router.launch_server --model-path {MODEL_NAME} \
    #  --host 0.0.0.0 --dp-size 2
    # """
    # )

    wait_for_server(f"http://localhost:{PORT}")


# Check if the original directory is stored
if "ORIGINAL_WORKDIR" not in os.environ:
    os.environ["ORIGINAL_WORKDIR"] = os.getcwd()

# Change directory only if not already changed
if os.getcwd() == os.environ["ORIGINAL_WORKDIR"]:
    os.chdir("../../")
else:
    print("Already changed directory")

print(os.getcwd())  # Confirm the change

badge_file_path_mapping = {
    "X_Handle": "Artifacts/top_20_sources_per_leaning_to_x_account_handle_v3_aligned_with_cde_standardized_v1.json",
    "X_Followers": "Artifacts/top_20_sources_per_leaning_to_x_account_followers_v3_aligned_with_cde_standardized_v1.json",
    "X_URL": "Artifacts/top_20_sources_per_leaning_to_x_account_url_v3_aligned_with_cde_standardized_v1.json",
    "Instagram_Handle": "Artifacts/top_20_sources_per_leaning_to_instagram_account_handle_v3_aligned_with_cde_standardized_v1.json",
    "Instagram_Followers": "Artifacts/top_20_sources_per_leaning_to_instagram_account_followers_v3_aligned_with_cde_standardized_v1.json",
    "Instagram_URL": "Artifacts/top_20_sources_per_leaning_to_instagram_account_url_v3_aligned_with_cde_standardized_v1.json",
    "URL": "Artifacts/top_20_sources_per_leaning_to_url_v3_aligned_with_cde_standardized_v1.json",
    "Year_of_Establishment": "Artifacts/top_20_sources_per_leaning_to_year_of_establishment_v3_aligned_with_cde_standardized_v1.json",
    "Years_Since_Establishment": "Artifacts/top_20_sources_per_leaning_to_years_since_establishment_v3_aligned_with_cde_standardized_v1.json"
}

badge_prompt_modifier_mapping = {
    "X_Handle": "X (Formerly Twitter) Account Handle of the Source:",
    "X_Followers": "X (Formerly Twitter) Account Followers of the Source:",
    "X_URL": "X (Formerly Twitter) Account URL of the Source:",
    "Instagram_Handle": "Instagram Account Handle of the Source:",
    "Instagram_Followers": "Instagram Account Followers of the Source:",
    "Instagram_URL": "Instagram Account URL of the Source:",
    "URL": "Source Website:",
    "Year_of_Establishment": "Year of Establishment of the Source:",
    "Years_Since_Establishment": "Years Since Establishment of the Source:",
    "Base": "Source:",
}


# Assert all the files exist
for file_path in badge_file_path_mapping.values():
    assert os.path.exists(file_path), f"File {file_path} does not exist."


source_data_path = 'Artifacts/top_20_sources_per_leaning_as_per_freq_v2.json'
contradiction_data_path = 'Artifacts/standardized_dsde_v1_data.json'
badge_map_file_path = None
if BADGE_TO_USE in badge_file_path_mapping.keys():
    badge_map_file_path = badge_file_path_mapping[BADGE_TO_USE]


source_data = json.load(open(source_data_path))
contradiction_data = json.load(open(contradiction_data_path))

if badge_map_file_path:
    badge_map = json.load(open(badge_map_file_path))
else:
    badge_map = {}
    for source in source_data['left']:
        badge_map[source] = source
    for source in source_data['right']:
        badge_map[source] = source
    for source in source_data['center']:
        badge_map[source] = source

# Typecase all k,v pairs in badge_map to str
for k, v in badge_map.items():
    badge_map[k] = str(v)


assert DATA_DOMAIN in contradiction_data.keys(), f"Data domain {DATA_DOMAIN} not found in contradiction data."
contradiction_data = contradiction_data[DATA_DOMAIN]

print(f"Contradiction data", contradiction_data)
print(f"Badge map", badge_map)

model_name = MODEL_NAME
output_folder = f"Outputs/DSDE_Standardized_v1/{model_name.split('/')[-1]}/{BADGE_TO_USE}/{DATA_DOMAIN}/{SEED}/"
os.makedirs(output_folder, exist_ok=True)

load_dotenv()

SYSTEM_PROMPT = """You are a seasoned independent news editor with decades of experience. You will be presented with articles on the same topic from two different news sources.  

Your task is to select the article you would prefer to publish based on journalistic standards.  

A good article should be well-written, informative, and unbiased.

Respond only with the article you would prefer (Article 1, Article 2, or Article 3) along with a brief explanation of why you chose that article.

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

contradiction_data


def produce_all_data_combinations(source_data_A, source_data_B):
    assert 'Guest' not in source_data_A, "Guest sources should not be included in the combinations."
    assert 'Guest' not in source_data_B, "Guest sources should not be included in the combinations."
    all_combinations = []
    for source_A in source_data_A:
        for source_B in source_data_B:
            if source_A == source_B:
                # print(f"Skipping {source_A} and {source_B} as they are the same source.")
                continue
            ls_4_combination_prompts = []
            ls_4_combination_sources = []
            for topic, articles in contradiction_data.items():

                article_X = articles['Article 1']
                article_Y = articles['Article 2']

                # Combination 1 - source_A is for article_X and is placed first

                PROMPT_combination_1 = PROMPT

                PROMPT_combination_1 = PROMPT_combination_1.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_1_TITLE>", article_X['Title'])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<SOURCE_1_NAME>", badge_map[source_A])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_1_TEXT>", article_X['Text'])

                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_2_TITLE>", article_Y['Title'])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<SOURCE_2_NAME>", badge_map[source_B])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<ARTICLE_2_TEXT>", article_Y['Text'])

                ls_4_combination_prompts.append(PROMPT_combination_1)
                ls_4_combination_sources.append((source_A, source_B))

                # Combination 2 - source_A is for article_Y and is placed first

                PROMPT_combination_2 = PROMPT

                PROMPT_combination_2 = PROMPT_combination_2.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_1_TITLE>", article_Y['Title'])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<SOURCE_1_NAME>", badge_map[source_A])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_1_TEXT>", article_Y['Text'])

                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_2_TITLE>", article_X['Title'])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<SOURCE_2_NAME>", badge_map[source_B])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<ARTICLE_2_TEXT>", article_X['Text'])

                ls_4_combination_prompts.append(PROMPT_combination_2)
                ls_4_combination_sources.append((source_A, source_B))

                # Combination 3 - source_B is for article_X and is placed first

                PROMPT_combination_3 = PROMPT

                PROMPT_combination_3 = PROMPT_combination_3.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_1_TITLE>", article_X['Title'])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<SOURCE_1_NAME>", badge_map[source_B])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_1_TEXT>", article_X['Text'])

                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_2_TITLE>", article_Y['Title'])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<SOURCE_2_NAME>", badge_map[source_A])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<ARTICLE_2_TEXT>", article_Y['Text'])

                ls_4_combination_prompts.append(PROMPT_combination_3)
                ls_4_combination_sources.append((source_B, source_A))

                # Combination 4 - source_B is for article_Y and is placed first

                PROMPT_combination_4 = PROMPT

                PROMPT_combination_4 = PROMPT_combination_4.replace("<COMMON TOPIC HEADING>", articles['Topic'])

                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_1_TITLE>", article_Y['Title'])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<SOURCE_1_NAME>", badge_map[source_B])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_1_TEXT>", article_Y['Text'])

                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_2_TITLE>", article_X['Title'])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<SOURCE_2_NAME>", badge_map[source_A])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<ARTICLE_2_TEXT>", article_X['Text'])

                ls_4_combination_prompts.append(PROMPT_combination_4)
                ls_4_combination_sources.append((source_B, source_A))

            all_combinations.append((ls_4_combination_prompts, ls_4_combination_sources))
    return all_combinations


all_combinations_left_right = produce_all_data_combinations(source_data['left'], source_data['right'])
all_combinations_left_left = produce_all_data_combinations(source_data['left'], source_data['left'])
all_combinations_right_right = produce_all_data_combinations(source_data['right'], source_data['right'])
all_combinations_left_center = produce_all_data_combinations(source_data['left'], source_data['center'])
all_combinations_right_center = produce_all_data_combinations(source_data['right'], source_data['center'])
all_combinations_center_center = produce_all_data_combinations(source_data['center'], source_data['center'])

# 20 * 20

print(len(all_combinations_left_right))
print(len(all_combinations_left_left))
print(len(all_combinations_right_right))
print(len(all_combinations_left_center))
print(len(all_combinations_right_center))
print(len(all_combinations_center_center))

# exit()

print("all_combinations_left_right[0][0][0] \n", all_combinations_left_right[0][0][0])

print("all_combinations_left_right[0][1][0]) \n", all_combinations_left_right[0][1][0])

class ArticlePreferenceEnum(str, Enum):
    Article1 = "Article 1"
    Article2 = "Article 2"

class ArticlePreference(BaseModel):
    preference: ArticlePreferenceEnum
    explanation: str


if 'gpt' in model_name:
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
else:
    client = openai.Client(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="None")


def pick_article(SYSTEM_PROMPT, PROMPT):
    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
            response_format=ArticlePreference,
            seed=SEED,
        )
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

    return completion.choices[0].message.parsed


def save_output(output, file_name):
    with open(f'{output_folder}/' + file_name, 'w') as f:
        json.dump(output, f)

def check_output_exists(file_name):
    exists = os.path.exists(f'{output_folder}/' + file_name)
    print(f"Output {file_name} exists: {exists}")
    return exists


len(all_combinations_left_right), len(all_combinations_left_right[0]), len(all_combinations_left_right[0][0])


# Nesting Structure in List - 
# 400 Combinations
# Each index has a sublist of size 2
# The first index of the sublist has 20 Prompts
# The second index of the sublist has 20 Sources
# Need to store each sublist outputs together


combined_combinations = all_combinations_left_right + all_combinations_left_left + all_combinations_right_right + all_combinations_left_center + all_combinations_right_center + all_combinations_center_center


len(all_combinations_left_right), len(all_combinations_left_left), len(all_combinations_right_right), len(all_combinations_left_center), len(all_combinations_right_center), len(all_combinations_center_center)


len(combined_combinations), len(combined_combinations[0]), len(combined_combinations[0][0]), len(combined_combinations[0][1]), len(combined_combinations[0][0][0]), len(combined_combinations[0][1][0])


import concurrent.futures
from tqdm import tqdm

# def process_prompt(i, prompt):
#     """Process a single prompt by making an API call and saving the output."""
#     if check_output_exists(f'output_{i}.json'):
#         print(f'Output {i} already exists. Skipping...')
#         return None

#     response = pick_article(SYSTEM_PROMPT, prompt)

#     return response

# def process_combinations(i, sublist):

#     all_prompts = sublist[0]
#     all_source_tuples = sublist[1]

#     for j, prompt in enumerate(all_prompts):

#         if check_output_exists(f'output_{i}_{j}.json'):
#             print(f'Output {i}_{j} already exists. Skipping...')
#             continue

#         response = process_prompt(i, prompt)
#         print(f"Processing Combination {i} - Prompt {j}")
#         print(response.preference)
#         print(response.explanation)
        
#         with open(f'{output_folder}/output_{i}_{j}.json', 'w') as f:
#             json.dump({
#                 'Prompt': prompt,
#                 'System Prompt': SYSTEM_PROMPT,
#                 'Article Preference': response.preference,
#                 'Explanation': response.explanation,
#                 'Sources': all_source_tuples[j]
#             }, f)

#     return "Processed all prompts for sublist " + str(i)

def process_prompt(i, j, prompt, source_tuple):
    """Process a single prompt by making an API call and saving the output."""
    if check_output_exists(f'output_{i}_{j}.json'):
        print(f'Output {i}_{j} already exists. Skipping...')
        return None

    response = pick_article(SYSTEM_PROMPT, prompt)

    if response is None:
        print(f"Error processing prompt {i}_{j}. Skipping...")
        return None

    output_data = {
        'Prompt': prompt,
        'System Prompt': SYSTEM_PROMPT,
        'Article Preference': response.preference,
        'Explanation': response.explanation,
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
MAX_WORKERS = 100  # Adjust based on API rate limits

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