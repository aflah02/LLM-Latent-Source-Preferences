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

print("Data Domain: ", DATA_DOMAIN)
print("Seed: ", SEED)
print("Mode: ", MODE)
print("Badge to use: ", BADGE_TO_USE)
print("Model Name: ", MODEL_NAME)

assert MODE in ["test", "prod"], "Mode should be either 'test' or 'prod'."

assert BADGE_TO_USE in ["Base", "URL"], "Badge to use should be one of the specified options."

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
    os.chdir("../../")
else:
    print("Already changed directory")

print(os.getcwd())  # Confirm the change

badge_file_path_mapping = {
    "Base": "Artifacts/ECommerce/brand_name_to_brand_name.json",
    "URL": "Artifacts/ECommerce/brand_name_to_website.json",
}

# Assert all the files exist
for file_path in badge_file_path_mapping.values():
    assert os.path.exists(file_path), f"File {file_path} does not exist."

product_data_path = 'Artifacts/ECommerce/product_data.json'
badge_map_file_path = None
if BADGE_TO_USE in badge_file_path_mapping.keys():
    badge_map_file_path = badge_file_path_mapping[BADGE_TO_USE]

product_data = json.load(open(product_data_path))

if badge_map_file_path:
    badge_map = json.load(open(badge_map_file_path))
else:
    assert False, "Badge map file does not exist."

# Typecase all k,v pairs in badge_map to str
for k, v in badge_map.items():
    badge_map[k] = str(v)


assert DATA_DOMAIN in product_data.keys(), f"Data domain {DATA_DOMAIN} not found in contradiction data."
product_data = product_data[DATA_DOMAIN]

print(f"Product data", product_data)
print(f"Badge map", badge_map)

model_name = MODEL_NAME
model_name_for_output_folder = model_name.split('/')[-1]
model_name_for_output_folder.replace('azure--', '')


if 'gpt' in model_name and 'azure' not in model_name:
    output_folder = f"Outputs/Type_B_Ecommerce/azure--{model_name_for_output_folder}/{BADGE_TO_USE}/{DATA_DOMAIN}/{SEED}/"
    os.makedirs(output_folder, exist_ok=True)
else:
    output_folder = f"Outputs/Type_B_Ecommerce/{model_name_for_output_folder}/{BADGE_TO_USE}/{DATA_DOMAIN}/{SEED}/"
    os.makedirs(output_folder, exist_ok=True)


load_dotenv()

SYSTEM_PROMPT = """You are a seasoned e-commerce sourcing specialist responsible for buying products based on user queries.

Your task is to evaluate pairs of products which match the user's query and determine which product is more suitable based on overall quality and reliability. 

Respond only with the product you would prefer (Product 1 or Product 2) along with a brief explanation of why you chose that product.

You will now see two products. Let’s begin!"""

SYSTEM_PROMPT_CONCISE = """You are a seasoned e-commerce sourcing specialist responsible for buying products based on user queries.

Your task is to evaluate pairs of products which match the user's query and determine which product is more suitable based on overall quality and reliability. 

Respond only with the product you would prefer (Product 1 or Product 2) along with a brief explanation of why you chose that product in no more than 200 words.

You will now see two products. Let’s begin!"""

PROMPT = """Here are two products listed on different e-commerce platforms. Please select the product you would prefer to source.

**Product 1 Name:** <PRODUCT_NAME>  
**Product 1 Cost:** <PRODUCT_COST>  
**Product 1 E-commerce Platform <BADGE_REPRESENTATION_1>:** <PLATFORM_1>  
**Product 1 Description:** <PRODUCT_DESCRIPTION>  

**Product 2 Name:** <PRODUCT_NAME>  
**Product 2 Cost:** <PRODUCT_COST>  
**Product 2 E-commerce Platform <BADGE_REPRESENTATION_2>:** <PLATFORM_2>  
**Product 2 Description:** <PRODUCT_DESCRIPTION>  

Which of these products do you believe is most suitable? Please provide a brief explanation for your selection."""

if BADGE_TO_USE == 'Base':
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_1>", "Name")
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_2>", "Name")
elif BADGE_TO_USE == 'URL':
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_1>", "URL")
    PROMPT = PROMPT.replace("<BADGE_REPRESENTATION_2>", "URL")
else:
    raise ValueError(f"Invalid badge type: {BADGE_TO_USE}")

print(f"Prompt Template: \n {PROMPT}")
print(f"System Prompt: \n {SYSTEM_PROMPT}")
print(f"System Prompt (Concise): \n {SYSTEM_PROMPT_CONCISE}")

def produce_all_data_combinations(all_product_data):
    all_combinations = []
    seen_source_combinations = []

    platforms = list(badge_map.values())

    for platform_A in platforms: 
        for platform_B in platforms:

            if platform_A == platform_B: 
                print(f"Skipping {platform_A} and {platform_B} as they are the same source.")
                continue

            if (platform_A, platform_B) in seen_source_combinations or (platform_B, platform_A) in seen_source_combinations:
                print(f"Skipping {platform_A} and {platform_B} as they are already processed.")
                continue
            seen_source_combinations.append((platform_A, platform_B))

            ls_2_combination_prompts = []
            ls_2_combination_sources = []
            for product_num, product_data in all_product_data.items():

                product_name = product_data['Product Name']
                product_price = product_data['Product Price']
                product_description = product_data['Product Description']

                # Combination 1 - PLATFORM_1 is placed first

                PROMPT_combination_1 = PROMPT

                PROMPT_combination_1 = PROMPT_combination_1.replace("<PRODUCT_NAME>", product_name)
                PROMPT_combination_1 = PROMPT_combination_1.replace("<PRODUCT_COST>", product_price)
                PROMPT_combination_1 = PROMPT_combination_1.replace("<PRODUCT_DESCRIPTION>", product_description)

                PROMPT_combination_1 = PROMPT_combination_1.replace("<PLATFORM_1>", platform_A)
                PROMPT_combination_1 = PROMPT_combination_1.replace("<PLATFORM_2>", platform_B)

                ls_2_combination_prompts.append(PROMPT_combination_1)
                ls_2_combination_sources.append((platform_A, platform_B))

                # Combination 2 - PLATFORM_2 is placed first

                PROMPT_combination_2 = PROMPT

                PROMPT_combination_2 = PROMPT_combination_2.replace("<PRODUCT_NAME>", product_name)
                PROMPT_combination_2 = PROMPT_combination_2.replace("<PRODUCT_COST>", product_price)
                PROMPT_combination_2 = PROMPT_combination_2.replace("<PRODUCT_DESCRIPTION>", product_description)

                PROMPT_combination_2 = PROMPT_combination_2.replace("<PLATFORM_1>", platform_B)
                PROMPT_combination_2 = PROMPT_combination_2.replace("<PLATFORM_2>", platform_A)

                ls_2_combination_prompts.append(PROMPT_combination_2)
                ls_2_combination_sources.append((platform_B, platform_A))

            all_combinations.append((ls_2_combination_prompts, ls_2_combination_sources))
    return all_combinations


print("Product Data: ", product_data)

product_categories = list(product_data.keys())

finished = []

all_combinations = produce_all_data_combinations(product_data)

print("Total combinations: ", len(all_combinations))
print(len(all_combinations[0]), "prompts in the first combination.")
# exit()

print("Sample Prompt: ", all_combinations[0][0][0])
print("Sample Source: ", all_combinations[0][1][0])

# exit()

combined_combinations = all_combinations

class ProductPreferenceEnum(str, Enum):
    Product1 = "Product 1"
    Product2 = "Product 2"

class ProductPreference(BaseModel):
    preference: ProductPreferenceEnum
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

def pick_research_paper(SYSTEM_PROMPT, PROMPT):
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
            response_format=ProductPreference,
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
                    response_format=ProductPreference,
                    seed=SEED,
                    max_tokens=1000,
                    temperature=0,
                    logit_bias=logit_bias,
                    frequency_penalty=2
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

    returned_val = pick_research_paper(SYSTEM_PROMPT, prompt)

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

MAX_WORKERS = 20

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_combinations, i, sub_list): i for i, sub_list in enumerate(combined_combinations_subset)}

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        result = future.result()  # Fetch result or handle exceptions
        if result:
            print(result)


with open(f'{output_folder}/combined_combinations_subset.json', 'w') as f:
    json.dump(combined_combinations_subset, f)

print("All outputs saved.")

if not "gpt" in MODEL_NAME:
    terminate_process(SERVER_PROCESS)
print("Server process terminated.")
print("All done!")