AI‑Generated Text Detection Across Models and Languages This repository
contains the code and supplementary materials for the Masters thesis
"Detection of AI‑Generated Text Across Models and Languages." The goal
of the project is to develop robust methods for distinguishing between
human‑authored and machine‑generated text in multiple languages. While
most prior work focuses solely on English, this project expands the
scope to English, Spanish and French. Thesis overview Modern language
models such as GPT‑3.5, Mistral 7B and Google's Gemini Pro can generate
fluent text that is difficult to distinguish from human writing. To
enable reliable detection across languages and generation models, we:
Collected a balanced corpus of 24 000 texts. Human documents were drawn
from reputable sources such as Wikipedia and BBC News, while
AI‑generated documents were created using GPT‑3.5‑Turbo‑0125, Mistral 7B
and Gemini Pro via two prompt styles: Topic: generate text based on a
given title and domain (e.g., Wikipedia or BBC). Continue: generate text
by continuing the first few sentences of a human document. Analysed
linguistic differences between human and AI texts, exploring readability
metrics, lexical diversity, syntactic complexity and repetitiveness.
Trained detection models using the multilingual transformer XLM‑RoBERTa.
We evaluated the system with three experimental setups: Cross‑model:
train on AI text from one model and test on another within the same
language. Cross‑lingual: train on one language and test on another.
Multi‑model multi‑lingual: combine all languages and models for
training. Achieved high accuracy and F1 scores across all setups. The
XLM‑RoBERTa model generalised well to unseen languages and generation
models, demonstrating the feasibility of multi‑lingual AI‑text
detection. If you would like to read the full thesis, a PDF is provided
in the repository
(32801688‑AmirMohammadi‑RezaHaffari‑TrangVu‑Thesis.pdf). Repository
structure ├── Code │ ├── 1_Prepare_Dataset \# Scripts/notebooks for
human‑text collection and preprocessing │ ├── 2_Gen_Ai_Text \# Scripts
and notebooks to generate AI‑authored text for each language │ │ ├──
English/Code \# Python scripts (e.g., Gen_Text_Mistral.py) and notebooks
for English │ │ ├── Spanish/Code \# Scripts/notebooks for Spanish │ │
├── French/Code \# Scripts/notebooks for French │ │ └── Persian/Code \#
Additional materials for Persian (not part of the thesis experiments) │
├── 3_Final_Dataset \# Notebooks for compiling the final dataset of
24 000 samples │ ├── 4_Feature_Extraction \# Notebooks/scripts for
analysing linguistic features and extracting statistics │ └── 5_Models │
├── Cross_Lingual_XLM-RoBERTa \# Cross-lingual experiments: train on one
language and test on another │ ├── Cross_Model_XLM-RoBERTa \#
Cross-model experiments: train on one AI generator and test on another │
├── Deepfake_Text_Detect │ │ ├── training \# Training scripts for the
XLM‑RoBERTa detection model │ │ └── deployment \# Utilities for
deploying a trained model │ └── Multi_Model_Multi_Lingual_XLM-RoBERTa \#
Multi-model multi-lingual experiments and final results ├── figures \#
(Optional) Plots and illustrations used in the thesis ├── thesis \#
Thesis PDF and related documents └── README.md (this file) Key files and
folders: Setup Clone the repository. If you only have access to a zip
file, extract it to your working directory. Create a Python environment
(e.g., using venv or Conda) with Python 3.10+. Install dependencies.
Recommended packages include: pip install torch transformers pandas tqdm
scikit-learn sentencepiece matplotlib seaborn Note: some notebooks also
require jupyter or ipykernel to run inside Jupyter. Obtain API
credentials for the language models (if necessary). The scripts assume
you have access to the respective APIs or local checkpoints for
Gemini Pro, Mistral 7B and GPT‑3.5. You may need to modify the code to
point to local models or your API keys. Generating the dataset Prepare
human texts. Use the notebooks under Code/1_Prepare_Dataset to download
and clean articles from the MIRACL Corpus and XL‑Sum dataset. Generate
AI texts. Run the scripts in Code/2_Gen_Ai_Text/`<language>`{=html}/Code
to produce AI‑authored documents. For example: python
Code/2_Gen_Ai_Text/English/Code/Gen_Text_Mistral.py\
--input_csv path/to/english_human_texts.csv\
--output_csv path/to/english_mistral_generated.csv\
--model_id mistral-7b --prompt_type Topic Each script defines two
functions: generate_ai_text_topic (generates a new text from a title and
domain) and generate_ai_text_continue (continues an existing text using
a random number of leading words). 3. Compile the dataset. Use
Code/3_Final_Dataset/Nine_Dataset.ipynb to merge the human and AI texts
into a single DataFrame with labels (human/AI), language, domain and
model ID. Training the detection model Navigate to the appropriate
folder under Code/5_Models depending on the experiment you wish to run:
Cross_Lingual_XLM-RoBERTa -- training scripts and notebooks for
cross‑lingual experiments (e.g., train on English and test on
Spanish/French). Cross_Model_XLM-RoBERTa -- training scripts and
notebooks for cross‑model experiments (train on one LLM's outputs, test
on another's). Multi_Model_Multi_Lingual_XLM-RoBERTa -- scripts for
training a single detector on the combined multi‑model multi‑lingual
dataset. Deepfake_Text_Detect/training -- baseline experiments using the
Deepfake/MAGE architecture. Inside this folder there is a longformer
subfolder containing main.py and a simple train.sh wrapper for launching
training. Each training script loads the relevant dataset, tokenises the
text using the XLM‑RoBERTa tokenizer (or Longformer for the Deepfake
baseline) and trains a classifier using PyTorch. Adjust hyperparameters
(batch size, epochs, learning rate) as needed. Feel free to explore
cross‑model, cross‑lingual and multi‑model multi‑lingual setups as
described in the thesis. Evaluate the model on held‑out test sets and
report accuracy/F1 scores. Evaluation results for the multi‑model
multi‑lingual experiment are provided in
multi_model_multi_lingual_evaluation_results.csv. See the thesis for
baseline numbers and ROC curves to compare against. Results and future
work Our experiments demonstrate that a multilingual transformer like
XLM‑RoBERTa can accurately distinguish human and AI texts across
languages and generation models. The system generalises well in transfer
scenarios, suggesting that style‑based differences are consistent across
languages. Future directions include: Extending the dataset to
additional languages (e.g., German, Italian, Persian). Exploring
lightweight models for on‑device detection. Investigating adversarial
robustness, where generation models attempt to evade detectors. Baseline
comparison: MAGE (formerly DeepFake) To benchmark our XLM‑RoBERTa
detector against other methods, we also evaluated the MAGE model
(formerly known as DeepFake Text Detection). MAGE is a machine‑generated
text detector originally released by Zhang et al. (2024). In our thesis
we trained MAGE on the multi‑model multi‑lingual dataset and compared
its performance with our own model using the same train/test splits. The
raw evaluation output for our XLM‑RoBERTa system is stored in
multi_model_multi_lingual_evaluation_results.csv, and analogous results
were produced for MAGE. To reproduce this comparison, clone the MAGE
repository () and adapt its training scripts to our dataset. We modified
its detect function to support multiple decision‑making approaches, as
discussed in the thesis. Licensing and citation This repository is
provided for academic use only. If you use this code or dataset in your
work, please cite the thesis and acknowledge the authors. You may also
include a license file of your choice (e.g., MIT or Apache 2.0) to
clarify usage rights.
