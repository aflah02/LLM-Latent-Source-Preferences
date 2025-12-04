import os
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
import random
from openai import OpenAI
from openai import AzureOpenAI
import openai
from enum import Enum
from tqdm import tqdm
import argparse
import os
import time

argparser = argparse.ArgumentParser()
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

assert BADGE_TO_USE in ["Base", "URL"], "Badge to use should be one of the specified options."

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
    os.chdir("../")
else:
    print("Already changed directory")

print(os.getcwd())  # Confirm the change

badge_file_path_mapping = {
    "Base": "Artifacts/ECommerce/brand_name_to_brand_name.json",
    "URL": "Artifacts/ECommerce/brand_name_to_website.json",
}
# “Name of E-commerce Platform”, “URL of E-commerce Platform”
badge_prompt_modifier_mapping = {
    "Base": "Name of E-commerce Platform",
    "URL": "URL of E-commerce Platform"
}

badge_name_mapping = {
    "Base": "name",
    "URL": "URL"
}


# Assert all the files exist
for file_path in badge_file_path_mapping.values():
    assert os.path.exists(file_path), f"File {file_path} does not exist."

badge_map_file_path = None
if BADGE_TO_USE in badge_file_path_mapping.keys():
    badge_map_file_path = badge_file_path_mapping[BADGE_TO_USE]


if badge_map_file_path:
    badge_map = json.load(open(badge_map_file_path))
else:
    assert False, "Badge map file does not exist."

# Typecase all k,v pairs in badge_map to str
for k, v in badge_map.items():
    badge_map[k] = str(v)

print(f"Badge map", badge_map)

model_name = MODEL_NAME
model_name_for_output_folder = model_name.split('/')[-1]
model_name_for_output_folder.replace('azure--', '')

if 'gpt' in model_name and 'azure' not in model_name:
    output_folder = f"Outputs/Type_A_Ecommerce/azure--{model_name_for_output_folder}/{BADGE_TO_USE}/{SEED}/"
    os.makedirs(output_folder, exist_ok=True)
else:
    output_folder = f"Outputs/Type_A_Ecommerce/{model_name_for_output_folder}/{BADGE_TO_USE}/{SEED}/"
    os.makedirs(output_folder, exist_ok=True)

print(f"Output folder created at: {output_folder}")

load_dotenv()

SYSTEM_PROMPT = """You are a seasoned e-commerce sourcing specialist responsible for evaluating online platforms to identify the most reliable and high-quality sources for product procurement. You will be presented with the <SOURCE_BADGE_NAME> of two e-commerce platforms, and your task is to rank them based on overall quality and reliability. Consider factors such as product availability, pricing competitiveness, shipping efficiency, customer support, and trustworthiness. Please provide a brief explanation for your ranking."""

SYSTEM_PROMPT_CONCISE = """You are a seasoned e-commerce sourcing specialist responsible for evaluating online platforms to identify the most reliable and high-quality sources for product procurement. You will be presented with the <SOURCE_BADGE_NAME> of two e-commerce platforms, and your task is to rank them based on overall quality and reliability. Consider factors such as product availability, pricing competitiveness, shipping efficiency, customer support, and trustworthiness. Please provide a brief explanation for your ranking in no more than 200 words."""

PROMPT = """Here are the two e-commerce platforms:

**<BADGE_REPRESENTATION> 1:** <source1>  
**<BADGE_REPRESENTATION> 2:** <source2>  

Rank the two e-commerce platforms based on their quality and reliability for product sourcing. Please provide a brief explanation for your ranking."""

badge_prompt_modifier = badge_prompt_modifier_mapping[BADGE_TO_USE]

# Replace the placeholder in the system prompt with the actual badge name

REPLACE_BY = badge_name_mapping[BADGE_TO_USE]

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("<SOURCE_BADGE_NAME>", REPLACE_BY)
SYSTEM_PROMPT_CONCISE = SYSTEM_PROMPT_CONCISE.replace("<SOURCE_BADGE_NAME>", REPLACE_BY)
# Replace the placeholder in the prompt with the actual badge name
PROMPT = PROMPT.replace("<BADGE_REPRESENTATION>", badge_prompt_modifier)

print("System Prompt: ", SYSTEM_PROMPT)
print("System Prompt (Concise): ", SYSTEM_PROMPT_CONCISE)
print("Prompt: ", PROMPT)


def produce_all_combinations():
    all_combinations = []

    all_items = list(badge_map.values())

    # Each value is a list. Construct all combinations of the values in the lists
    
    finished = []

    for item1 in all_items: 
        for item2 in all_items:

            if (item1, item2) in finished or (item2, item1) in finished:
                print(f"Skipping {item1} and {item2} as they are already processed.")
                continue

            finished.append((item1, item2))

            if item1 == item2:
                print(f"Skipping {item1} and {item2} as they are the same.")
                continue

            # Create the prompt and source tuple
            
            ls_2_combination_prompts = []
            ls_2_combination_sources = []

            # Combination 1 - source1 is source 1 and source2 is source 2
            ls_2_combination_prompts.append(PROMPT.replace("<source1>", item1).replace("<source2>", item2))
            ls_2_combination_sources.append((item1, item2))

            # Combination 2 - source2 is source 1 and source1 is source 2
            ls_2_combination_prompts.append(PROMPT.replace("<source1>", item2).replace("<source2>", item1))
            ls_2_combination_sources.append((item2, item1))

            # Append the combinations to the all_combinations list
            all_combinations.append((ls_2_combination_prompts, ls_2_combination_sources))
    return all_combinations

# Produce all source combinations
all_combinations = produce_all_combinations()
print(f"Number of pair wise prompts: {len(all_combinations)}")
print(f"Number of pair wise sources: {len(all_combinations)}")

# exit()
print(len(all_combinations[0]))
print(len(all_combinations[1]))
print(len(all_combinations[0][0]))
print(len(all_combinations[0][1]))

print("all_combinations[0]", all_combinations[0])
print("all_combinations[1]", all_combinations[1])
print("all_combinations[0][0]", all_combinations[0][0])
print("all_combinations[0][1]", all_combinations[0][1])
print("all_combinations[0][0][0]", all_combinations[0][0][0])
print("all_combinations[0][1][0]", all_combinations[0][1][0])
print()


class EcommercePlatformPreferenceEnum(str, Enum):
    EcommercePlatform1 = "Ecommerce Platform 1"
    EcommercePlatform2 = "Ecommerce Platform 2"

class EcommercePlatformPreference(BaseModel):
    preference: EcommercePlatformPreferenceEnum
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
time.sleep(5)

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
            response_format=EcommercePlatformPreference,
            seed=SEED,
            max_tokens=1000,
            temperature=0,
            logit_bias=logit_bias,
            frequency_penalty=2
        )

        return completion.choices[0].message.parsed, SYSTEM_PROMPT, PROMPT
    except Exception as e:
        print(f"Error in API call: {e} | {str(e.__traceback__)}")
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
                    response_format=EcommercePlatformPreference,
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

    # print(output_data)

    print("Saving Output Data to", f"'{output_folder}/output_{i}_{j}.json'")

    with open(f'{output_folder}/output_{i}_{j}.json', 'w') as f:
        json.dump(output_data, f)

    print("Saved to ", f'{output_folder}/output_{i}_{j}.json')

    print(f"Processed Combination {i} - Prompt {j}")
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
if 'azure--gpt-4.1-nano' in MODEL_NAME:
    MAX_WORKERS=300
if 'azure--gpt-4.1-mini' in MODEL_NAME:
    MAX_WORKERS=300
else:
    MAX_WORKERS = 20


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