# %%
import os
from os import path
import pandas as pd
import numpy as np
import glob
from pqdm.threads import pqdm


# %% [markdown]
# # Definitions

# %%
HOME=os.path.expanduser('~')
LIFE2SCENARIO_ROOT_PATH=path.join(HOME,"Documents/life2scenario_core/datasets/life2scenario_medium")
DATASET_ROOT_PATH=path.join(LIFE2SCENARIO_ROOT_PATH,"train")
print(DATASET_ROOT_PATH)

# %%
PROMPTS_ROOT=path.join(DATASET_ROOT_PATH, "prompts")
REFERENCE_ROOT=path.join(DATASET_ROOT_PATH, "ref_scenarios")
TARGET_ROOT=path.join(DATASET_ROOT_PATH, "target_scenarios")

# %%
PREP_PICKLES_ROOT=path.join(LIFE2SCENARIO_ROOT_PATH, "prep_pickles")

# %% [markdown]
# # Utils

# %%
def readFile(file):
    cur_target = open(file).read()
    return [file, cur_target]
    
def load_file(filename: str):
    return np.load(filename)

def save_np_to_file(data: np.ndarray, filename: str):
    np.save(filename, data)
    print(f"Saved to {filename}")

def save_pd_to_file(data: pd.DataFrame, filename: str):
    data.to_csv(filename, index=False)  

# %%
def file2index(filename: str):
    return os.path.basename(filename).split(".")[0].split("_")[-1]

def file_data_pairs_to_data_dict(in_arr: np.ndarray):
    return {
                "data": [data for data in in_arr[:, 1]],
                "id": [file2index(filename) for filename in in_arr[:, 0]]
            }


# %% [markdown]
# # Prepare DataFrame

# # %%
# prompt_arr = []
# prompt_file_list = glob.glob(f"{PROMPTS_ROOT}/*.txt", recursive=False)
# prompt_read_results = pqdm(prompt_file_list, readFile, n_jobs=64)
# prompt_arr = np.array(prompt_read_results)

# print(f"{len(prompt_arr)} data points will be saved.")
# save_np_to_file(prompt_arr, path.join(PREP_PICKLES_ROOT, "prompt_arr.npy"))

# # %%
# ref_arr = []
# ref_file_list = glob.glob(f"{REFERENCE_ROOT}/*.xosc", recursive=False)
# ref_read_results = pqdm(ref_file_list, readFile, n_jobs=64)
# ref_arr = np.array(ref_read_results)

# print(f"{len(ref_arr)} data points will be saved.")
# save_np_to_file(ref_arr, path.join(PREP_PICKLES_ROOT, "ref_arr.npy"))

# # %%
# target_arr = []
# target_file_list = glob.glob(f"{TARGET_ROOT}/*.xosc", recursive=False)
# target_read_results = pqdm(target_file_list, readFile, n_jobs=64)
# target_arr = np.array(target_read_results)
# len(target_arr)

# %%
df_train = pd.DataFrame()

# %% [markdown]
# ## Load Array Pickles

# %%
target_arr = load_file(path.join(PREP_PICKLES_ROOT, "target_arr.npy"))
ref_arr = load_file(path.join(PREP_PICKLES_ROOT, "ref_arr.npy"))
prompt_arr = load_file(path.join(PREP_PICKLES_ROOT, "prompt_arr.npy"))

# %%
target_dict = file_data_pairs_to_data_dict(target_arr)
ref_dict = file_data_pairs_to_data_dict(ref_arr)
prompt_dict = file_data_pairs_to_data_dict(prompt_arr)

# %%
target_df = pd.DataFrame(target_dict)
target_df = target_df.rename(columns={"data": "target_scenario"})

ref_df = pd.DataFrame(ref_dict)
ref_df = ref_df.rename(columns={"data": "reference_scenario"})

prompt_df = pd.DataFrame(prompt_dict)
prompt_df = prompt_df.rename(columns={"data": "prompt"})

# %%
target_prompt_df = pd.merge(target_df, prompt_df, on="id")
train_df = pd.merge(target_prompt_df, ref_df, on="id")
train_df.head()

# %%
train_df.describe()

# %% [markdown]
# ## Format-like `Stack Exchange Instruction @ HuggingFace`

# %%
train_df["request"] = train_df[['prompt', 'reference_scenario']].apply(lambda x : '{}?\n```\n{}\n```'.format(x[0], x[1]), axis=1)
train_df["response"] = train_df['target_scenario'].apply(lambda x : 'Here is the result:\n```\n{}\n```'.format(x))

# %%
print(train_df["request"][1])

# %%
print(train_df["response"][1])

# %%
train_df.head()

# %% [markdown]
# ## Save the DataFrame

# %%
save_pd_to_file(train_df, path.join(PREP_PICKLES_ROOT, "train_dataset.csv"))

# %% [markdown]
# # Create HuggingFace Dataset

# %%
from datasets import load_dataset
from datasets import Dataset

# %% [markdown]
# ## Load Train DataFrame

# %%
train_final = pd.read_csv(path.join(PREP_PICKLES_ROOT, "train_dataset.csv"))

train_final = train_final[["request", "response"]]
train_final.head()



