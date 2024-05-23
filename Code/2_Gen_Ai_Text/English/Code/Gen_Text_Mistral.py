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

    # Determine the user message based on the domain
    if language == "English":
        if domain == "Wikipedia":
            user_message = f"<s> You are a Wikipedia writer. I need only text, without a title. It must be between 50 to 100 words. The response must be exclusively in English based on the title: '{title}'. </s>"
        elif domain == "BBC":
            user_message = f"<s> You are a BBC news writer. I need only text, without a title. It must be between 400 to 700 words. The response must be exclusively in English based on the title: '{title}'. </s>"
        else:
            user_message = "<s> Error: Domain not supported. </s>"
    elif language == "Spanish":
        if domain == "Wikipedia":
            user_message = f"<s> Eres un escritor de Wikipedia. Solo necesito texto, sin título. Debe ser entre 50 y 100 palabras. La respuesta debe ser exclusivamente en español basada en el título: '{title}'. </s>"
        elif domain == "BBC":
            user_message = f"<s> Eres un redactor de noticias de la BBC. Solo necesito texto, sin título. Debe ser entre 400 y 700 palabras. La respuesta debe ser exclusivamente en español basada en el título: '{title}'. </s>"
        else:
            user_message = "<s> Error: Dominio no admitido. </s>"
    elif language == "French":
        if domain == "Wikipedia":
            user_message = f"<s> Vous êtes un rédacteur de Wikipedia. Je n'ai besoin que de texte, sans titre. Il doit être entre 50 et 100 mots. La réponse doit être exclusivement en français sur le titre : '{title}'. </s>"
        elif domain == "BBC":
            user_message = f"<s> Vous êtes un rédacteur de nouvelles pour la BBC. Je n'ai besoin que de texte, sans titre. Il doit être entre 400 et 700 mots. La réponse doit être exclusivement en français sur le titre : '{title}'. </s>"
        else:
            user_message = "<s> Erreur : Domaine non pris en charge. </s>"
    else:
        user_message = "<s> Error: Language not supported. </s>"

    # Construct the conversation context
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""}
    ]

    # Prepare the input for the model
    inputs = tokenizer.apply_chat_template(
        messages, padding=True, return_tensors="pt").to(device)

    # Generate the response
    outputs = model.generate(
        inputs, max_new_tokens=1200, do_sample=True)

    # Decode the generated text
    text_content = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-processing to clean up the text
    text_content = text_content.replace("\n\n", "\n").replace("\n", " ")
    text_content = re.sub(r"\[INST\].*?\[/INST\]", "",
                          text_content, flags=re.DOTALL)

    return text_content

# ------------------------------------------------------------

def generate_ai_text_continue(title, text, language, domain, model, tokenizer, device):

    """Generate text by AI based on a randomly chosen number of words (between 20 and 40) from the text for the given title, language, and domain."""
    # Split the text into words, grab the first 40 words, then randomly choose a number between 20-40
    # Finally, select that amount of words from the text
    num_words = random.randint(20, 40)
    text_start = ' '.join(text.split()[:40][:num_words])


    # Determine the user message based on the domain and language
    if language == "English":
        if domain == "Wikipedia":
            user_message = f"<s> You are a Wikipedia writer. I need only text, without a title. It must be between 50 to 100 words. The response must be exclusively in English on the topic '{title}' after the following text: '{text_start}'. </s>"
        elif domain == "BBC":
            user_message = f"<s> You are a BBC news writer. I need only text, without a title. It must be between 400 to 700 words. The response must be exclusively in English on the topic '{title}' after the following text: '{text_start}'. </s>"
        else:
            user_message = "Error: Domain not supported."
    elif language == "Spanish":
        if domain == "Wikipedia":
            user_message = f"<s> Eres un escritor de Wikipedia. Solo necesito texto, sin título. Debe ser entre 50 y 100 palabras. La respuesta debe ser exclusivamente en español sobre el tema '{title}' después del siguiente texto: '{text_start}'. </s>"
        elif domain == "BBC":
            user_message = f"<s> Eres un redactor de noticias de la BBC. Solo necesito texto, sin título. Debe ser entre 400 y 700 palabras. La respuesta debe ser exclusivamente en español sobre el tema '{title}' después del siguiente texto: '{text_start}'. </s>"
        else:
            user_message = "<s> Error: Dominio no admitido. </s>"
    elif language == "French":
        if domain == "Wikipedia":
            user_message = f"<s> Vous êtes un rédacteur de Wikipedia. Je n'ai besoin que de texte, sans titre. Il doit être entre 50 et 100 mots. La réponse doit être exclusivement en français sur le sujet '{title}' après le texte suivant : '{text_start}'. </s>"
        elif domain == "BBC":
            user_message = f"<s> Vous êtes un rédacteur de nouvelles pour la BBC. Je n'ai besoin que de texte, sans titre. Il doit être entre 400 et 700 mots. La réponse doit être exclusivement en français sur le sujet '{title}' après le texte suivant : '{text_start}'. </s>"
        else:
            user_message = "<s> Erreur : Domaine non pris en charge. </s>"
    else:
        user_message = "<s> Error: Language not supported. </s>"

    # Construct the conversation context
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""}
    ]

    # Prepare the input for the model
    inputs = tokenizer.apply_chat_template(
        messages, padding=True, return_tensors="pt").to(device)

    # Generate the response
    outputs = model.generate(
        inputs, max_new_tokens=1200, do_sample=True)

    # Decode the generated text
    text_content = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-processing to clean up the text
    text_content = text_content.replace("\n\n", "\n").replace("\n", " ")
    text_content = re.sub(r"\[INST\].*?\[/INST\]", "",
                          text_content, flags=re.DOTALL)

    return text_content


# ------------------------------------------------------------


def update_dataset_with_ai_text(dataset, model, model_id, method, tokenizer, device):
    """Update the dataset with AI-generated text based on the method."""
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Generating AI Text"):
        if row['language'] in ['English', 'Spanish', 'French']:
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
    
    file_path = 'XLSum_es_test.csv'

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    method = "Topic"

    # Check if CUDA (GPU support) is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, device_map="auto").to(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_id).to(device)
    
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

