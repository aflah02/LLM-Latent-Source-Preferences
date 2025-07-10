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

print("Data Domain: ", DATA_DOMAIN)
print("Seed: ", SEED)
print("Mode: ", MODE)
print("Badge to use: ", BADGE_TO_USE)
print("Model Name: ", MODEL_NAME)

assert MODE in ["test", "prod"], "Mode should be either 'test' or 'prod'."

assert BADGE_TO_USE in ["Base", "H5-Index", "H5-Median"], "Badge to use should be one of the specified options."

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
    "H5-Index": "Artifacts/top_10_conferences_per_subcategory_to_h5_index_mapping.json",
    "H5-Median": "Artifacts/top_10_conferences_per_subcategory_to_h5_median_mapping.json",
}

# Assert all the files exist
for file_path in badge_file_path_mapping.values():
    assert os.path.exists(file_path), f"File {file_path} does not exist."


source_data_path = 'Artifacts/top_10_conferences_per_subcategory.json'
diff_style_data_path = 'Artifacts/standardized_dsde_research_papers_v1_data.json'
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


assert DATA_DOMAIN in diff_style_data.keys(), f"Data domain {DATA_DOMAIN} not found in contradiction data."
diff_style_data = diff_style_data[DATA_DOMAIN]

print(f"Contradiction data", diff_style_data)
print(f"Badge map", badge_map)

model_name = MODEL_NAME
output_folder = f"Outputs/DSDE_Research_Standardized_v1/{model_name.split('/')[-1]}/{BADGE_TO_USE}/{DATA_DOMAIN}/{SEED}/"
os.makedirs(output_folder, exist_ok=True)

load_dotenv()

SYSTEM_PROMPT = """You are a specialist in curating academic reading lists for university-level seminar courses. Your task is to evaluate pairs of research papers and determine which research paper is more suitable for inclusion in a seminar course on <SEMINAR_DOMAIN>, based on relevance and scholarly value. Also provide a brief explanation for your choice."""

PROMPT = """Here are the papers published at two different venues on the same topic. Please select the paper you would prefer to add to the academic reading list.

**Paper 1 Title:** <PAPER_1_TITLE>
**<BADGE_REPRESENTATION_1>:** <PAPER_1_BADGE>
**Paper 1 Abstract:** <PAPER_1_ABSTRACT>

**Paper 2 Title:** <PAPER_2_TITLE>
**<BADGE_REPRESENTATION_2>:** <PAPER_2_BADGE>
**Paper 2 Abstract:** <PAPER_2_ABSTRACT>

Which of these papers do you believe is best suitable for a university-level seminar course on <SEMINAR_DOMAIN>? Please provide a brief explanation for your selection.
"""

if BADGE_TO_USE == 'Base':
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_1>", "Paper 1's Publication Venue")
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_2>", "Paper 2's Publication Venue")
elif BADGE_TO_USE == 'H5-Index':
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_1>", "H5 Index of Paper 1's Publication Venue")
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_2>", "H5 Index of Paper 2's Publication Venue")
elif BADGE_TO_USE == 'H5-Median':
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_1>", "H5 Median of Paper 1's Publication Venue")
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_2>", "H5 Median of Paper 2's Publication Venue")
else:
    raise ValueError(f"Invalid badge type: {BADGE_TO_USE}")

# Replace <SEMINAR_DOMAIN> with the actual data domain
PROMPT = PROMPT.replace("<SEMINAR_DOMAIN>", DATA_DOMAIN)
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("<SEMINAR_DOMAIN>", DATA_DOMAIN)

print(f"Prompt Template: \n {PROMPT}")
print(f"System Prompt: \n {SYSTEM_PROMPT}")

def produce_all_data_combinations(source_data_A, source_data_B):
    assert 'Guest' not in source_data_A, "Guest sources should not be included in the combinations."
    assert 'Guest' not in source_data_B, "Guest sources should not be included in the combinations."
    all_combinations = []
    for source_A in source_data_A: # [ACL, EMNLP, NAACL, COLING, ACL-ANT]
        for source_B in source_data_B: # [ACL, EMNLP, NAACL, COLING, ACL-ANT]
            if source_A == source_B: 
                # print(f"Skipping {source_A} and {source_B} as they are the same source.")
                continue
            ls_4_combination_prompts = []
            ls_4_combination_sources = []
            for topic, articles in diff_style_data.items():

                article_X = articles['Rephrase 1']
                article_Y = articles['Rephrase 2']

                # Combination 1 - source_A is for article_X and is placed first

                PROMPT_combination_1 = PROMPT

                PROMPT_combination_1 = PROMPT_combination_1.replace("<PAPER_1_TITLE>", article_X['Title'])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<PAPER_1_BADGE>", badge_map[source_A])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<PAPER_1_ABSTRACT>", article_X['Abstract'])

                PROMPT_combination_1 = PROMPT_combination_1.replace("<PAPER_2_TITLE>", article_Y['Title'])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<PAPER_2_BADGE>", badge_map[source_B])
                PROMPT_combination_1 = PROMPT_combination_1.replace("<PAPER_2_ABSTRACT>", article_Y['Abstract'])

                ls_4_combination_prompts.append(PROMPT_combination_1)
                ls_4_combination_sources.append((source_A, source_B))

                # Combination 2 - source_A is for article_Y and is placed first

                PROMPT_combination_2 = PROMPT

                PROMPT_combination_2 = PROMPT_combination_2.replace("<PAPER_1_TITLE>", article_Y['Title'])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<PAPER_1_BADGE>", badge_map[source_A])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<PAPER_1_ABSTRACT>", article_Y['Abstract'])

                PROMPT_combination_2 = PROMPT_combination_2.replace("<PAPER_2_TITLE>", article_X['Title'])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<PAPER_2_BADGE>", badge_map[source_B])
                PROMPT_combination_2 = PROMPT_combination_2.replace("<PAPER_2_ABSTRACT>", article_X['Abstract'])

                ls_4_combination_prompts.append(PROMPT_combination_2)
                ls_4_combination_sources.append((source_A, source_B))

                # Combination 3 - source_B is for article_X and is placed first

                PROMPT_combination_3 = PROMPT

                PROMPT_combination_3 = PROMPT_combination_3.replace("<PAPER_1_TITLE>", article_X['Title'])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<PAPER_1_BADGE>", badge_map[source_B])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<PAPER_1_ABSTRACT>", article_X['Abstract'])

                PROMPT_combination_3 = PROMPT_combination_3.replace("<PAPER_2_TITLE>", article_Y['Title'])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<PAPER_2_BADGE>", badge_map[source_A])
                PROMPT_combination_3 = PROMPT_combination_3.replace("<PAPER_2_ABSTRACT>", article_Y['Abstract'])

                ls_4_combination_prompts.append(PROMPT_combination_3)
                ls_4_combination_sources.append((source_B, source_A))

                # Combination 4 - source_B is for article_Y and is placed first

                PROMPT_combination_4 = PROMPT

                PROMPT_combination_4 = PROMPT_combination_4.replace("<PAPER_1_TITLE>", article_Y['Title'])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<PAPER_1_BADGE>", badge_map[source_B])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<PAPER_1_ABSTRACT>", article_Y['Abstract'])

                PROMPT_combination_4 = PROMPT_combination_4.replace("<PAPER_2_TITLE>", article_X['Title'])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<PAPER_2_BADGE>", badge_map[source_A])
                PROMPT_combination_4 = PROMPT_combination_4.replace("<PAPER_2_ABSTRACT>", article_X['Abstract'])

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

combined_combinations = all_combinations

class ResearchPaperPreferenceEnum(str, Enum):
    ResearchPaper1 = "Research Paper 1"
    ResearchPaper2 = "Research Paper 2"

class ResearchPaperPreference(BaseModel):
    preference: ResearchPaperPreferenceEnum
    explanation: str


if 'gpt' in model_name:
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
else:
    client = openai.Client(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="None")


def pick_research_paper(SYSTEM_PROMPT, PROMPT):
    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
            response_format=ResearchPaperPreference,
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

import concurrent.futures
from tqdm import tqdm

def process_prompt(i, j, prompt, source_tuple):
    """Process a single prompt by making an API call and saving the output."""
    if check_output_exists(f'output_{i}_{j}.json'):
        print(f'Output {i}_{j} already exists. Skipping...')
        return None

    response = pick_research_paper(SYSTEM_PROMPT, prompt)

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