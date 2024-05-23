from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import pandas as pd
from tqdm.auto import tqdm
import random
import re
# ------------------------------------------------------------


def read_dataset(file_path):
    """Read the dataset from a CSV file."""
    return pd.read_csv(file_path, encoding='utf-8')

# ------------------------------------------------------------

def generate_ai_text_topic(title, language, domain, model, tokenizer, device):
    """Generate text by AI based on the title, language, and domain."""

    TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
    # Determine the user message based on the domain
    if language == "Persian":
        if domain == "Wikipedia":
            CONTEXT = "This is a conversation with PersianMind. You can ask it anything you want and " \
            "Here PersianMind acts like a Wikipedia author. PersianMind will do its best to give you an accurate and relevant information."
            PROMPT = f"یک پاراگراف درباره '{title}' بنویس. فقط متن لازم است، بدون نیاز به عنوان."
            max_new_tokens = 140  # Set max_new_tokens for Wikipedia
        elif domain == "BBC":
            CONTEXT = "This is a conversation with PersianMind. You can ask it anything you want and " \
            "Here PersianMind acts like a News author. PersianMind will do its best to give you an accurate and relevant information."
            PROMPT = f"یک پاراگراف درباره '{title}' بنویس. فقط متن لازم است، بدون نیاز به عنوان."
            max_new_tokens = 140  # Set max_new_tokens for BBC
        else:
            CONTEXT = "Error: Domain not supported."
            max_new_tokens = 0  
    else:
        CONTEXT = "Error: Language not supported."
        max_new_tokens = 0  

    # Construct the conversation context
    messages = TEMPLATE.format(context=CONTEXT, prompt=PROMPT)

    # Prepare the input for the model
    inputs = tokenizer(messages, padding=True, return_tensors="pt").to(device)

    # Generate the response
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

    # Decode the generated text
    text_content = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Post-processing to clean up the text
    text_content = text_content.replace("\n\n", "\n").replace("\n", " ")

    return text_content[len(messages):]


# ------------------------------------------------------------

def generate_ai_text_continue(title, text, language, domain, model, tokenizer, device):

    """Generate text by AI based on a randomly chosen number of words (between 20 and 40) from the text for the given title, language, and domain."""
    # Split the text into words, grab the first 40 words, then randomly choose a number between 20-40
    # Finally, select that amount of words from the text
    num_words = random.randint(20, 40)
    text_start = ' '.join(text.split()[:40][:num_words])

    TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
    # Determine the user message based on the domain and language
    if language == "Persian":
        if domain == "Wikipedia":
            CONTEXT = "This is a conversation with PersianMind. You can ask it anything you want and " \
            "Here PersianMind acts like a Wikipedia author. PersianMind will do its best to give you an accurate and relevant information."
            PROMPT = f"درباره '{title}' عنوان ادامه '{text_start}' جمله را کامل کن و توضیح بده."
            max_new_tokens = 140  # Set max_new_tokens for Wikipedia
        elif domain == "BBC":
            CONTEXT = "This is a conversation with PersianMind. You can ask it anything you want and " \
            "Here PersianMind acts like a News author. PersianMind will do its best to give you an accurate and relevant information."
            PROMPT = f"درباره '{title}' عنوان ادامه '{text_start}' جمله را کامل کن و توضیح بده."
            max_new_tokens = 140  # Set max_new_tokens for BBC
        else:
            CONTEXT = "Error: Domain not supported."
            max_new_tokens = 0
    else:
        CONTEXT = "Error: Language not supported."
        max_new_tokens = 0

    # Construct the conversation context
    messages = TEMPLATE.format(context=CONTEXT, prompt=PROMPT)

    # Prepare the input for the model
    inputs = tokenizer(messages, padding=True, return_tensors="pt").to(device)

    # Generate the response
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

    # Decode the generated text
    # text_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text_content = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Post-processing to clean up the text
    text_content = text_content.replace("\n\n", "\n").replace("\n", " ")

    return text_content[len(messages):]

# ------------------------------------------------------------

def update_dataset_with_ai_text(dataset, model, model_id, method, tokenizer, device):
    """Update the dataset with AI-generated text based on the method."""
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Generating AI Text"):
        if row['language'] in ['Persian']:
            if method == "Continue":
                ai_text = generate_ai_text_continue(row['title'], row['text'], row['language'], row['domain'], model, tokenizer, device)
            elif method == "Topic":
                ai_text = generate_ai_text_topic(row['title'], row['language'], row['domain'], model, tokenizer, device)
            else:
                raise ValueError("Unsupported method. Choose either 'Continue' or 'Topic'.")
            
            dataset.at[index, 'text'] = ai_text
            dataset.at[index, 'write_by'] = 'Ai'
            dataset.at[index, 'language'] = row['language']
            dataset.at[index, 'domain'] = row['domain']
            dataset.at[index, 'method'] = method
            dataset.at[index, 'LLM_model'] = model_id
            dataset.at[index, 'label'] = 0
    return dataset

# ------------------------------------------------------------

def save_new_dataset(dataset, new_file_path):
    """Save the updated dataset to a new CSV file."""
    dataset.to_csv(new_file_path, index=False, encoding='utf-8')

# ------------------------------------------------------------

if __name__ == "__main__":
    
    file_path = 'BBC_Persian_Human_Topic.csv'

    model_id = "universitytehran/PersianMind-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    method = "Topic"

    # Check if CUDA (GPU support) is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, device_map="auto").to(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
    ).to(device)
    
    dataset = read_dataset(file_path)

    updated_dataset = update_dataset_with_ai_text(
        dataset, model, model_id, method, tokenizer, device)

    # Grabbing unique values for filename
    domain = updated_dataset['domain'].unique()[0]
    language = updated_dataset['language'].unique()[0]
    write_by = updated_dataset['write_by'].unique()[0]
    method = updated_dataset['method'].unique()[0]
    llm_model = updated_dataset['LLM_model'].unique()[0]

    # Replace forward slashes in the model_id with underscores
    safe_model_id = model_id.replace("/", "_")

    # Construct the new filename using the safe_model_id
    new_file_name = f"{domain}_{language}_{write_by}_{method}_({safe_model_id}).csv"

    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the new filename to create an absolute path
    new_file_path = os.path.join(script_directory, new_file_name)

    # Saving the updated dataset
    save_new_dataset(updated_dataset, new_file_path)


