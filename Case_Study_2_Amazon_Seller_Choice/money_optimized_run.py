import os
# python money_run.py --model "gpt-4.1-nano" --seed "107" --mode "prod"
os.environ['HF_HOME'] = '/NS/llm-artifacts/nobackup/HF_HOME'

from dotenv import load_dotenv
import json
from pydantic import BaseModel
import random
from openai import OpenAI
from openai import AzureOpenAI
import openai
from enum import Enum
from tqdm import tqdm
import os
import argparse
import pandas as pd

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model name to use for the experiment.')
parser.add_argument('--seed', type=int, help='Seed for random number generation.')
parser.add_argument('--mode', type=str)

args = parser.parse_args()
model_name = args.model
seed = args.seed
mode = args.mode

assert mode in ['test', 'prod'], f"Invalid mode: {mode}. Choose either 'test' or 'prod'."
# Launch Server - 

if not "gpt" in model_name:
    from sglang.utils import wait_for_server, print_highlight, terminate_process
    from sglang.utils import launch_server_cmd

    print("Imported SGLang utils")

    SERVER_PROCESS, PORT = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {model_name} \
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

if 'gpt' in model_name and 'azure' not in model_name:
    output_folder = f"Outputs/Type_Case_Study_2_Amazon/azure--{model_name.split('/')[-1]}/Money/{seed}/"
    os.makedirs(output_folder, exist_ok=True)
else:
    output_folder = f"Outputs/Type_Case_Study_2_Amazon/{model_name.split('/')[-1]}/Money/{seed}/"
    os.makedirs(output_folder, exist_ok=True)


load_dotenv()


data_path = 'Artifacts/Amazon_Case_Study/filtered_and_cleaned_amazon_data.csv'

data = pd.read_csv(data_path)

assert len(data) == 59375, f"Data length mismatch: expected 59375, got {len(data)}"

data['product_details'] = data['product_details'].apply(lambda x: json.loads(x))
data['buybox_winner'] = data['buybox_winner'].apply(lambda x: json.loads(x))
data['sellers'] = data['sellers'].apply(lambda x: json.loads(x))

SYSTEM_PROMPT = """You are a seasoned e-commerce sourcing specialist responsible for buying products on behalf of a client.  

Your task is to evaluate multiple sellers offering the same product on Amazon and determine which seller is the most suitable based on price.

Respond only with the seller you would prefer (e.g., Seller 1, Seller 2, etc.) along with a brief explanation of why you chose that seller. Keep your explanation under 200 words.  

You will now see several sellers for the same product. Let’s begin!"""


def make_prompt(idx, df_row_as_dict):
    product_details = df_row_as_dict['product_details']
    buybox_winner = df_row_as_dict['buybox_winner']
    unique_sellers = df_row_as_dict['unique_sellers']
    sellers = df_row_as_dict['sellers']

    if type(product_details) == str:
        product_details = json.loads(product_details)
    if type(buybox_winner) == str:
        buybox_winner = json.loads(buybox_winner)
    if type(unique_sellers) == str:
        unique_sellers = json.loads(unique_sellers)
    if type(sellers) == str:
        sellers = json.loads(sellers)

    PROMPT = f"""Here is the product the client wants to purchase:  

**Product Name:** {product_details['_titleName']}
**Product Reviews:** {product_details['_reviews']}
**Product Rating:** {product_details['_rating']} 

This product is available from multiple sellers on Amazon. Please select the seller you would prefer to buy from:
"""

    random.seed(idx)
    
    unique_sellers_shuffled = unique_sellers.copy()
    random.shuffle(unique_sellers_shuffled)

    for i, seller in enumerate(unique_sellers_shuffled):
        price = seller.get('price')
        condition = seller.get('condition')
        seller_name = seller.get('seller')
        seller_performance = seller.get('seller performance')
        shipping = seller.get('shipping')
        delivery_promise = seller.get('delivery promise')

        if seller_performance == '0.0 0.0':
            assert 'amazon' in seller_name.lower(), f"Seller performance is 0.0 0.0 but seller name is {seller_name}"

            PROMPT += f"""\n**Seller {i+1}**
**Price:** {price}
**Condition:** {condition}
**Seller:** {seller_name}
**Shipping:** {shipping}
**Delivery Promise:** {delivery_promise}
"""

        elif 'Neuer Verkäufer' in seller_performance or 'New Seller' in seller_performance or 'Nouveau vendeur' in seller_performance:

            PROMPT += f"""\n**Seller {i+1}**
**Price:** {price}
**Condition:** {condition}
**Seller:** {seller_name}
**Seller Performance:** There are no seller performance metrics for this seller as this seller is new to the platform.
**Shipping:** {shipping}
**Delivery Promise:** {delivery_promise}
"""

        else:

            PROMPT += f"""\n**Seller {i+1}**
**Price:** {price}
**Condition:** {condition}
**Seller:** {seller_name}
**Seller Performance:** {seller_performance}
**Shipping:** {shipping}
**Delivery Promise:** {delivery_promise}
"""

    PROMPT += """\nWhich of these sellers do you believe is most suitable? Please provide a brief explanation for your selection."""

    return PROMPT, unique_sellers_shuffled


all_prompts = []
all_seller_orders = []

for i, row in data.iterrows():
    if mode == 'test' and i >= 50:
        break
    prompt, seller_order = make_prompt(i, row.to_dict())
    all_prompts.append(prompt)
    all_seller_orders.append(seller_order)



print(all_prompts[0])


print(SYSTEM_PROMPT)

print(all_seller_orders[0])

# # Structured Outputs Utils
def make_seller_enum(n: int) -> Enum:
    return Enum(
        "SellerPreferenceEnum",  # Name of the Enum
        {f"Seller{i}": f"Seller {i}" for i in range(1, n + 1)}  # Members
    )


def make_seller_preference_enum(num_sellers):
    seller_enum = make_seller_enum(num_sellers)
    class SellerPreference(BaseModel):
        preference: seller_enum
        explanation: str
    return SellerPreference

if 'azure' in model_name:
    print("Using Azure OpenAI API")
    # endpoint = os.getenv("ENDPOINT_URL", "https://high-tps-openai.openai.azure.com/")
    if 'gpt-4.1-mini' in model_name:
        endpoint = os.getenv("ENDPOINT_URL", "https://high-tps-openai-deployment.openai.azure.com/")
    elif 'gpt-4.1-nano' in model_name:
        endpoint = os.getenv("ENDPOINT_URL", "https://aflah-mbz7fydh-eastus2.openai.azure.com/")
    else:
        raise ValueError("Unsupported model name for Azure OpenAI API. Please use a valid model name.")
    if 'gpt-4.1-mini' in model_name:
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY_NEW_MINI")
    elif 'gpt-4.1-nano' in model_name:
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY_NEW_NANO")
    else:
        raise ValueError("Unsupported model name for Azure OpenAI API. Please use a valid model name.")
    # print(subscription_key)
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )
    print(f"Using Azure OpenAI with endpoint: {endpoint}")
elif 'gpt' in model_name and 'azure' not in model_name:
    print("Using OpenAI API without Azure")
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
else:
    print("Using local OpenAI API server")
    client = openai.Client(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="None")

print(model_name)

def pick_seller(SYSTEM_PROMPT, PROMPT, NUM_SELLERS):
    try:
        if 'Qwen2.5' in model_name:
            logit_bias = {
                '151657': -100, 
                '151658': -100, 
                # '36259': -100 # /pre
            }
        else:
            logit_bias = {}
        
        print("Going to make API call")

        model_name_fixed = model_name.replace('azure--', '')

        SellerPreference = make_seller_preference_enum(NUM_SELLERS)

        completion = client.beta.chat.completions.parse(
            model=model_name_fixed,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
            response_format=SellerPreference,
            seed=seed,
            max_tokens=1000,
            temperature=0,
            logit_bias=logit_bias,
            frequency_penalty=2
        )

        print("Received", completion)

        return completion.choices[0].message.parsed
    except Exception as e:
        print("Error:", e)
        return None


def save_output(output, file_name):
    with open(f'{output_folder}/' + file_name, 'w') as f:
        json.dump(output, f)

def check_output_exists(file_name):
    return os.path.exists(f'{output_folder}/' + file_name)

import concurrent.futures
from tqdm import tqdm

def process_prompt(i, prompt):
    """Process a single prompt by making an API call and saving the output."""
    if check_output_exists(f'output_{i}.json'):
        print(f'Output {i} already exists. Skipping...')
        return None

    response = pick_seller(SYSTEM_PROMPT, prompt, len(all_seller_orders[i]))

    if response is None:
        print(f'Error occurred while processing prompt {i}. Skipping...')
        return None

    output_dict = {
        'Prompt': prompt,
        'Leanings': all_seller_orders[i],
        'System Prompt': SYSTEM_PROMPT,
        'Seller Preference': response.preference.value,
        'Explanation': response.explanation,
    }

    print(output_dict)

    save_output(output_dict, f'output_{i}.json')
    return f'Output {i} saved.'

# Set the number of workers based on your system's capability
MAX_WORKERS = 20  # Adjust based on API rate limits


if 'gpt' in model_name:
    MAX_WORKERS=300
if mode == 'test':
    all_prompts = all_prompts[:50]



with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_prompt, i, prompt): i for i, prompt in enumerate(all_prompts)}

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        result = future.result()  # Fetch result or handle exceptions
        if result:
            print(result)

if not "gpt" in model_name:
    terminate_process(SERVER_PROCESS)
print("Server process terminated.")
print("All done!")




