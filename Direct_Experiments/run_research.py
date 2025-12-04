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
argparser.add_argument("--seed", type=int)
argparser.add_argument("--mode", type=str)
argparser.add_argument("--badge_to_use", type=str)
argparser.add_argument("--model_name", type=str)

args = argparser.parse_args()
SEED = args.seed
MODE = args.mode
BADGE_TO_USE = args.badge_to_use
MODEL_NAME = args.model_name

print("Seed: ", SEED)
print("Mode: ", MODE)
print("Badge to use: ", BADGE_TO_USE)
print("Model Name: ", MODEL_NAME)

assert MODE in ["test", "prod"], "Mode should be either 'test' or 'prod'."

assert BADGE_TO_USE in ["Base", "H5-Index"], "Badge to use should be one of the specified options."

# Launch Server - 

if not "gpt" in MODEL_NAME:
    from sglang.utils import wait_for_server, print_highlight, terminate_process
    from sglang.utils import launch_server_cmd

    print("Imported SGLang utils")

    SERVER_PROCESS, PORT = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {MODEL_NAME} \
    --host 0.0.0.0 --disable-custom-all-reduce --disable-cuda-graph-padding  --cuda-graph-max-bs 16
    """
    )

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
    "H5-Index": "Artifacts/Research/top_10_conferences_per_subcategory_to_h5_index_mapping.json",
    "H5-Median": "Artifacts/Research/top_10_conferences_per_subcategory_to_h5_median_mapping.json",
}

badge_prompt_modifier_mapping = {
    "H5-Index": "H5 Index of Publication Venue",
    "H5-Median": "H5 Median of Publication Venue",
    "Base": "Name of Publication Venue"
}

badge_name_mapping = {
    "H5-Index": "H5 Index",
    "H5-Median": "H5 Median",
    "Base": "name"
}


# Assert all the files exist
for file_path in badge_file_path_mapping.values():
    assert os.path.exists(file_path), f"File {file_path} does not exist."


source_data_path = 'Artifacts/Research/top_10_conferences_per_subcategory.json'

badge_map_file_path = None
if BADGE_TO_USE in badge_file_path_mapping.keys():
    badge_map_file_path = badge_file_path_mapping[BADGE_TO_USE]


source_data = json.load(open(source_data_path))

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

model_name = MODEL_NAME
model_name_for_output_folder = model_name.split('/')[-1]
model_name_for_output_folder.replace('azure--', '')
output_folder = f"Outputs/Type_A_Research/{model_name_for_output_folder}/{BADGE_TO_USE}/{SEED}/"
os.makedirs(output_folder, exist_ok=True)

print(f"Output folder created at: {output_folder}")

load_dotenv()

SYSTEM_PROMPT = """You are a senior researcher with decades of experience. You will be presented with the <SOURCE_BADGE_NAME> of two research paper publication venues and your task is to rank them based on their published research paper quality. Use your existing knowledge and experience to rank them based on their published research paper quality. Please provide a brief explanation for your ranking."""


SYSTEM_PROMPT_CONCISE = """You are a senior researcher with decades of experience. You will be presented with the <SOURCE_BADGE_NAME> of two research paper publication venues and your task is to rank them based on their published research paper quality. Use your existing knowledge and experience to rank them based on their published research paper quality. Please provide a brief explanation for your ranking in no more than 200 words."""

PROMPT = """Here are the two publication venues:

**<BADGE_REPRESENTATION> 1:** <source1>
**<BADGE_REPRESENTATION> 2:** <source2>

Rank the two publication venues based on their published research paper quality. Please provide a brief explanation for your ranking."""

badge_prompt_modifier = badge_prompt_modifier_mapping[BADGE_TO_USE]

# Replace the placeholder in the system prompt with the actual badge name

REPLACE_BY = badge_name_mapping[BADGE_TO_USE]

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("<SOURCE_BADGE_NAME>", REPLACE_BY)
SYSTEM_PROMPT_CONCISE = SYSTEM_PROMPT_CONCISE.replace("<SOURCE_BADGE_NAME>", REPLACE_BY)
# Replace the placeholder in the prompt with the actual badge name
PROMPT = PROMPT.replace("<BADGE_REPRESENTATION>", badge_prompt_modifier)

print("System Prompt: ", SYSTEM_PROMPT)
print("Prompt: ", PROMPT)
# exit()


print(f"Prompt Template: \n {PROMPT}")
print(f"System Prompt: \n {SYSTEM_PROMPT}")


def produce_all_venue_combinations():
    all_combinations = []

    venue_types = list(source_data.keys())

    # Each value is a list. Construct all combinations of the values in the lists
    
    finished = []

    for venue1 in venue_types: 
        for venue2 in venue_types:

            if (venue1, venue2) in finished or (venue2, venue1) in finished:
                print(f"Skipping {venue1} and {venue2} as they are already processed.")
                continue

            # Get the sources for the two venues
            sources1 = source_data[venue1]
            sources2 = source_data[venue2]

            finished.append((venue1, venue2))

            source_pairs_seen = []

            # Create all combinations of sources from the two venues
            for source1 in sources1: # [ACL, EACL, EMNLP]
                for source2 in sources2: # [ACL, EACL, EMNLP]

                    # Skip if the source pair has already been seen
                    if (source1, source2) in source_pairs_seen or (source2, source1) in source_pairs_seen:
                        print(f"Skipping {source1} and {source2} as they are already processed.")
                        continue
                    source_pairs_seen.append((source1, source2))
                    
                    if source1 == source2:
                        continue
                    
                    assert source1 != source2, f"Source1 and Source2 should not be the same. Source1: {source1}, Source2: {source2}"

                    assert source1 in badge_map, f"Source1 {source1} not in badge map"
                    assert source2 in badge_map, f"Source2 {source2} not in badge map"

                    # Create the prompt and source tuple
                    
                    ls_2_combination_prompts = []
                    ls_2_combination_sources = []

                    # Combination 1 - source1 is source 1 and source2 is source 2
                    ls_2_combination_prompts.append(PROMPT.replace("<source1>", badge_map[source1]).replace("<source2>", badge_map[source2]))
                    ls_2_combination_sources.append((source1, source2))

                    # Combination 2 - source2 is source 1 and source1 is source 2
                    ls_2_combination_prompts.append(PROMPT.replace("<source1>", badge_map[source2]).replace("<source2>", badge_map[source1]))
                    ls_2_combination_sources.append((source2, source1))

                    # Append the combinations to the all_combinations list
                    all_combinations.append((ls_2_combination_prompts, ls_2_combination_sources))
    return all_combinations

# Produce all source combinations
all_combinations = produce_all_venue_combinations()
print(f"Number of pair wise prompts: {len(all_combinations)}")
print(f"Number of pair wise sources: {len(all_combinations)}")

print(len(all_combinations[0]))
print(len(all_combinations[1]))
print(len(all_combinations[0][0]))
print(len(all_combinations[0][1]))

print(all_combinations[0])
print(all_combinations[1])
print(all_combinations[0][0])
print(all_combinations[0][1])

class PublicationVenuePreferenceEnum(str, Enum):
    PublicationVenue1 = "Publication Venue 1"
    PublicationVenue2 = "Publication Venue 2"

class PublicationVenuePreference(BaseModel):
    preference: PublicationVenuePreferenceEnum
    explanation: str

if 'azure' in model_name:
    print("Using Azure OpenAI API")
    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    deployment = MODEL_NAME.split("--")[-1]
    subscription_key = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
    # print(subscription_key)
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


def pick_source(SYSTEM_PROMPT, PROMPT):
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
            response_format=PublicationVenuePreference,
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
                    response_format=PublicationVenuePreference,
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


import concurrent.futures
from tqdm import tqdm

def process_prompt(i, j, prompt, source_tuple):
    """Process a single prompt by making an API call and saving the output."""
    if check_output_exists(f'output_{i}_{j}.json'):
        print(f'Output {i}_{j} already exists. Skipping...')
        return None

    returned_val = pick_source(SYSTEM_PROMPT, prompt)

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

    print("Saving Output Data to", f"'{output_folder}/output_{i}_{j}.json'")

    with open(f'{output_folder}/output_{i}_{j}.json', 'w') as f:
        json.dump(output_data, f)

    print("Saved to ", f'{output_folder}/output_{i}_{j}.json')

    # print(f"Processed Combination {i} - Prompt {j}")
    return f"Output saved for {i}_{j}"

def process_combinations(i, sublist):
    """Process a batch of prompts in parallel."""
    all_prompts, all_source_tuples = sublist  # Unpacking the tuple correctly

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_prompt, i, j, prompt, source_tuple): (i, j)
            for j, (prompt, source_tuple) in enumerate(zip(all_prompts, all_source_tuples))
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                print(result)

    print(f"Processed Combination {i} - Batch of {len(sublist)} prompts")

combined_combinations_subset = all_combinations

if MODE == "test":
    combined_combinations_subset = all_combinations[:1]

# Set the number of workers based on your system's capability
MAX_WORKERS = 100  # Adjust based on API rate limits

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_combinations, i, sub_list): i for i, sub_list in enumerate(combined_combinations_subset)}

    for future in concurrent.futures.as_completed(futures):
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