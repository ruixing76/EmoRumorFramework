import os
import ast
import sys
import time
import json
import random
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import linregress
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import List

from tqdm import tqdm
from itertools import chain
from datetime import datetime
from collections import deque, Counter, OrderedDict

fig_dir = '../raw_data/fig'

EMO = ['anger', 'disgust', 'fear', 'sadness', 'pessimism', 'neutral', 'surprise', 'anticipation', 'trust', 'optimism', 'joy', 'love']
EMO_no_neutral = ['anger', 'disgust', 'fear', 'sadness', 'pessimism', 'surprise', 'anticipation', 'trust', 'optimism', 'joy', 'love']
EMO_agg= ['negative','neutral','positive']

NEG_EMO = ['anger', 'disgust', 'fear', 'sadness', 'pessimism']
POS_EMO = ['surprise', 'anticipation', 'trust', 'optimism', 'joy', 'love']
SENT = ['-3', '-2', '-1', '0', '1', '2', '3']
SENT_no_neutral = ['-3', '-2', '-1', '1', '2', '3']

MOST_COMMON = 5

###### Data IO ######
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, file_path, is_friendly_format=True, is_verbose=False):
    if is_friendly_format:
        indent = 4
    else:
        indent = None
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)
    if is_verbose:
        print(f"Data is saved to {file_path}")


def read_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return data


def write_file(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)


###### Utility Function ######
def print_emo_count(comments_emotion: List[str]):
    neg_cnt=0
    pos_cnt=0
    neu_cnt=0
    for emo,cnt in Counter(comments_emotion).items():
        # print(emo,cnt)
        if emo in NEG_EMO:
            neg_cnt+=cnt
        elif emo in POS_EMO:
            pos_cnt+=cnt
        else:
            neu_cnt+=cnt
    print(f"negative emotions: {neg_cnt}\npositive emotions: {pos_cnt}\nneutral: {neu_cnt}\n")

def normalize_emotion(emotion):
    if emotion in NEG_EMO:
        return 'negative'
    elif emotion in POS_EMO:
        return 'positive'
    else:
        return 'neutral'

def normalize_emotion_pairs(emotion_pairs:List):
    return [(normalize_emotion(from_emo),normalize_emotion(to_emo)) for from_emo,to_emo in emotion_pairs]

def preprocess(df):
    df['id'] = df['id'].astype(str)  # note the id is originally numpy64, but the structure id is string
    df['emollm_emotion'] = df['emollm_emotion'].astype(str)
    df['emollm_sentiment'] = df['emollm_sentiment'].str.split(':').str[0].astype(int)
    return df


# recursively collect all keys from the tree
def _collect_all_keys(tree):
    # print(tree.keys())
    if isinstance(tree, str):
        tree = ast.literal_eval(tree)
    keys = set()
    for key, subtree in tree.items():
        keys.add(key)
        if isinstance(subtree, dict):
            keys.update(_collect_all_keys(subtree))
        else:
            keys.update(subtree)
    return keys


def collect_keys(tree, return_type='all'):
    if isinstance(tree, str):
        tree = ast.literal_eval(tree)
    all_keys = _collect_all_keys(tree)
    root_key = list(tree.keys())[0]
    if return_type == 'all_comments':
        all_keys.remove(root_key)
        return all_keys
    elif return_type == 'direct_comments':
        direct_reply_keys = set(tree[root_key].keys())
        return direct_reply_keys
    elif return_type == 'all':
        return all_keys
    else:
        raise ValueError("return_type should be one of all, all_comments or direct_comments!")


# source tweet emotion distribution
def process_emotion(x, primary=False):
    emotions = []
    for each in x.replace('neutral or no emotion', 'neutral').replace(' ', ',').split(','):
        emotion = each.strip().replace('.', '').replace(';', '')
        if len(emotion) == 0 or emotion not in EMO:
            continue
        emotions.append(emotion)
    if len(emotions) >= 1 and primary is True:
        emotions = emotions[0]
    return emotions


def get_labeled(df, sample=False):
    rumour = df[df['rumour_type'] == 'rumours']
    non_rumour = df[df['rumour_type'] == 'non-rumours']
    if sample:
        sample_n = min(len(rumour), len(non_rumour))
        rumour = rumour.sample(sample_n)
        non_rumour = non_rumour.sample(sample_n)
    print(f"rumour size:{len(rumour)}")
    print(f"non rumour size:{len(non_rumour)}")
    return rumour, non_rumour


def order_counter(counter):
    ith_step = Counter(counter)
    ordered_dict = OrderedDict((key, 0) for key in EMO)
    for key, count in ith_step.items():
        if key in ordered_dict:
            ordered_dict[key] = count
    return ordered_dict


def format_time(string):
    date_format = "%a %b %d %H:%M:%S %z %Y"
    parsed_date = datetime.strptime(string, date_format)
    return parsed_date


###### Plot function ######
# pandas histplot wrapper
def sns_plot(df, bins='auto', xlim=None, ylim=None, title=None, xlabel=None, xlabels=None, size=(4, 3), auto=False,
             saveto=None):
    if auto == True:
        xlim = None
        ylim = None
    plt.figure(figsize=size)
    # plot figure according to given xlabels order
    if xlabels is not None:
        df = pd.DataFrame(df, columns=['target'])
        df['target'] = df['target'].astype(pd.CategoricalDtype(categories=xlabels, ordered=True))
        sns.histplot(data=df, x='target', kde=True, bins=bins)
    else:
        sns.histplot(data=df, kde=True, bins=bins)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if saveto is not None:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/{saveto}", bbox_inches='tight')
    plt.show()


def sns_plot2(df1, df2=None, bins='auto', df1_name='rumour', df2_name='non_rumour', font_size=16, xlim=None, ylim=None,
              title=None,
              xlabel=None, xlabels=None, size=(4, 3), auto=False, saveto=None):
    if auto:
        xlim, ylim = None, None

    sns.set_context("notebook", rc={"axes.titlesize": font_size, "axes.labelsize": font_size,
                                    "xtick.labelsize": font_size - 2, "ytick.labelsize": font_size - 2})

    plt.figure(figsize=size)

    # Ensure the data in both DataFrames aligns with the desired x-labels order
    if xlabels is not None:
        df1 = pd.DataFrame(df1, columns=['target'])
        df1['target'] = df1['target'].astype(pd.CategoricalDtype(categories=xlabels, ordered=True))

        if df2 is not None:
            df2 = pd.DataFrame(df2, columns=['target'])
            df2['target'] = df2['target'].astype(pd.CategoricalDtype(categories=xlabels, ordered=True))

    # Plot the histogram for the first DataFrame
    sns.histplot(data=df1, x='target', kde=True, bins=bins, color='blue', label=df1_name)

    # Plot the second DataFrame if provided
    if df2 is not None:
        sns.histplot(data=df2, x='target', kde=True, bins=bins, color='orange', label=df2_name, alpha=0.6)

    # Apply optional configurations
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)

    plt.legend(fontsize=font_size - 4)  # Add a legend to distinguish between DataFrames
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    if saveto:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/{saveto}", bbox_inches='tight')
    plt.show()


###### emotion transition ######
sentiment_scheme = 'emollm_sentiment'
emotion_scheme = 'emollm_emotion'


def get_factuality(df, sample=True):
    true = df[df['factuality'] == 'true']
    false = df[df['factuality'] == 'false']
    unverified = df[df['factuality'] == 'unverified']
    if sample:
        sample_n = min(len(true), len(false), len(unverified))
        true = true.sample(sample_n)
        false = false.sample(sample_n)
        unverified = unverified.sample(sample_n)
    print(f"true size:{len(true)}")
    print(f"false size:{len(false)}")
    print(f"unverified size:{len(unverified)}")
    return true, false, unverified


# extract pairwise emotions based on thread tree structure
def bfs_tree_pairwise(tree):
    if isinstance(tree, str):
        tree = ast.literal_eval(tree)

    # Result list where each sublist contains nodes at the same level
    result = []

    # List to store [parent id, child id] pairs where parent id are not root
    parent_child_pairs_non_root = []

    # List to store [root id, child id] pairs
    root_child_pairs = []

    # Queue for BFS, initialized with the root node and its level (0)
    queue = deque([(tree, 0)])

    while queue:
        # Dequeue a node and its level
        node, level = queue.popleft()

        # Ensure the result list has a sublist for the current level
        if len(result) <= level:
            result.append([])

        # Get the key and its children
        for key, children in node.items():
            # Append the current node to its level list
            result[level].append(key)

            # Enqueue children with incremented level
            for child in children:
                # For parent-child pairs where parent id is not root
                if level > 0:
                    parent_child_pairs_non_root.append([key, child])

                # For root-child pairs
                else:
                    root_child_pairs.append([key, child])
                queue.append(({child: node[key][child]}, level + 1))
    return result, parent_child_pairs_non_root, root_child_pairs


# extract pairwise emotions chronologically without thread structure
def timestamp_pairwise(ids):
    if len(ids) < 2:
        return []  # If there are less than 2 elements, return an empty list
    adjacent_pairs = []
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        adjacent_pairs.append(pair)
    return adjacent_pairs


# extract n_gram ids
def timestamp_n_gram(ids, n=3, fix_src=False):
    n_gram_pairs = []
    # Iterate through the emotions list to generate pairs
    if len(ids) < n:
        return n_gram_pairs
    for i in range(len(ids) - n + 1):
        # Extract the N-gram pair
        n_gram = ids[i:i + n]
        if fix_src:
            n_gram[0] = ids[0]

        # The first element is the previous (N-1) emotions joined as a tuple
        previous_n_1 = tuple(n_gram[:-1])

        # The second element is the Nth emotion
        nth_emotion = n_gram[-1]
        # Append the pair to the list
        n_gram_pairs.append((previous_n_1, nth_emotion))

    return n_gram_pairs


# test
# emotions = ['neutral', 'happy', 'joy', 'optimism', 'sadness','sadness']
# n = 3  # Specify the N for N-gram pairs

# n_gram_pairs = generate_n_gram_pairs(emotions, n)
# print(n_gram_pairs)

def get_aff_from_id(df, id_pairs, scheme=emotion_scheme):
    pair_emo_list = []
    for pair in id_pairs:
        # we assume only one emotion for each node
        parent_emo = tuple(df[df['id'] == index][scheme].iloc[0] for index in pair[0])
        child_emo = df[df['id'] == pair[1]][scheme].iloc[0]
        if len(parent_emo) == 0 or len(child_emo) == 0:
            continue
        pair_emo_list.append((parent_emo, child_emo))
    return pair_emo_list


def get_emo_from_id(df, id_pairs):
    pair_emo_list = []
    for pair in id_pairs:
        parent_emo = df[df['id'] == pair[0]][emotion_scheme]
        child_emo = df[df['id'] == pair[1]][emotion_scheme]
        if len(parent_emo) == 0 or len(child_emo) == 0:
            continue
        assert len(parent_emo) == 1
        assert len(child_emo) == 1
        parent_emo = parent_emo.iloc[0]
        child_emo = child_emo.iloc[0]
        pair_emo_list.append((parent_emo, child_emo))
    return pair_emo_list


def get_sent_from_id(df, id_pairs):
    pair_emo_list = []
    for pair in id_pairs:
        parent_emo = df[df['id'] == pair[0]][sentiment_scheme]
        child_emo = df[df['id'] == pair[1]][sentiment_scheme]
        if len(parent_emo) == 0 or len(child_emo) == 0:
            continue
        assert len(parent_emo) == 1
        assert len(child_emo) == 1
        parent_emo = parent_emo.iloc[0]
        child_emo = child_emo.iloc[0]
        pair_emo_list.append((parent_emo, child_emo))
    return pair_emo_list


# build emotion transition matrix
def build_transition_matrix(emotion_pairs, norm='row', aff='emo', no_neutral=True, is_log=True):
    if aff == 'emo':
        if no_neutral:
            emotions = EMO_no_neutral
        else:
            emotions = EMO
    elif aff =='emo_agg':
        emotions = EMO_agg
    else:
        if no_neutral:
            emotions = SENT_no_neutral
        else:
            emotions = SENT
    # Initialize transition matrix with zeros
    transition_matrix = np.zeros((len(emotions), len(emotions)), dtype=int)

    emotion_to_index = {emo: i for i, emo in enumerate(emotions)}

    # Count transitions in the emotion_pairs
    for pair in emotion_pairs:
        from_emotion, to_emotion = pair
        from_idx = emotion_to_index[from_emotion]
        to_idx = emotion_to_index[to_emotion]
        transition_matrix[from_idx, to_idx] += 1

    transition_matrix = transition_matrix + 1  # Add 1 smoothing
    if is_log:
        transition_matrix = np.log2(transition_matrix)
    # normalize transition matrix to obtain probabilities
    if norm == 'row':
        axis = 1
    elif norm == 'col':
        axis = 0
    else:
        axis = None
    sums = transition_matrix.sum(axis=axis, keepdims=True)
    transition_matrix_normalized = np.where(sums != 0, transition_matrix / sums, 0)
    return transition_matrix, transition_matrix_normalized


# Example list of pairs where each element is a list
def product_pairs(pairs):
    pure_pairs = [(emotion1, emotion2) for sublist1, sublist2 in pairs
                  for emotion1 in sublist1
                  for emotion2 in sublist2]
    return pure_pairs


def get_emo_pairs(data, pair_type='root_pairs_emo', no_neutral=True, n_sample=0):
    emo_pairs = data[data[pair_type].apply(len) != 0][pair_type].explode().to_list()
    if no_neutral:
        emo_pairs = [pair for pair in emo_pairs if
                     pair[0] and pair[1] and not 'neutral' in pair[0] and not 'neutral' in pair[1]]
    else:
        emo_pairs = [pair for pair in emo_pairs if pair[0] and pair[1]]
    if n_sample > 0:
        print(f'Sampling {n_sample} from pairs...')
        emo_pairs = random.sample(emo_pairs, n_sample)
    print(f"{pair_type} pair size: {len(emo_pairs)}")
    return emo_pairs


# no neutral now becomes a default option, can be merged with get_emo_pairs
def get_sent_pairs(data, pair_type='root_pairs_sent', n_sample=0):
    emo_pairs = data[data[pair_type].apply(len) != 0][pair_type].explode().to_list()
    emo_pairs = [pair for pair in emo_pairs if pair[0] and pair[1] and (pair[0] != '0' and pair[1] != '0')]
    if n_sample > 0:
        print(f'Sampling {n_sample} from pairs...')
        emo_pairs = random.sample(emo_pairs, n_sample)
    print(f"{pair_type} pair size: {len(emo_pairs)}")
    return emo_pairs


def plot_matrix(matrix, aff='emo', no_neutral=True, font_size=16, annot=None,
                vmax=None, vmin=None, fig_size=(14,8), title=None, saveto=None):
    # Determine emotions based on aff and no_neutral flags
    if aff == 'emo':
        emotions = EMO_no_neutral if no_neutral else EMO
    elif aff =='emo_agg':
        emotions = EMO_agg
    else:
        emotions = SENT_no_neutral if no_neutral else SENT

    # Plotting
    plt.figure(figsize=fig_size)
    annot = matrix if annot is None else annot
    tick_intervals = np.arange(vmin, vmax, 0.1)
    ax = sns.heatmap(matrix, annot=annot, cmap="coolwarm", fmt=".2f", vmax=vmax, vmin=vmin,
                     xticklabels=emotions, yticklabels=emotions, cbar=True,
                     cbar_kws={'ticks': tick_intervals},
                     annot_kws={"size": font_size - 3,})  # annotation font size

    # highlight high probable transitions
    # for text in ax.texts:
    #     value = float(text.get_text())
    #     if value >= 0.2:
    #         text.set_fontsize(font_size-1)  # Set font size
    #         text.set_weight("bold")  # Make text bold

    # Set font sizes for labels and title
    plt.title(title if title else 'Emotion Transition Matrix', fontsize=font_size + 2)
    # xlabel rotates the x-axis labels
    plt.xlabel('To Emotion', fontsize=font_size)
    plt.ylabel('From Emotion', fontsize=font_size)

    # Adjust tick label sizes
    plt.xticks(rotation=90, fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Save plot if saveto is provided
    if saveto is not None:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/{saveto}", bbox_inches='tight')

    plt.show()


# build and visualize the transition matrix
def create_matrix(emo_pairs, aff='emo', no_neutral=True, is_log=True, title=None, vmax=None, vmin=None):
    transition_matrix, transition_matrix_normalized = build_transition_matrix(emo_pairs, norm='row', aff=aff,
                                                                              no_neutral=no_neutral, is_log=is_log)
    plot_matrix(transition_matrix_normalized, aff=aff, no_neutral=no_neutral, font_size=20,
                vmax=vmax, vmin=vmin, title=title, saveto=f'{title}.pdf')
    # return transition_matrix,transition_matrix_normalized


def create_pure_matrix(emo_pairs, aff='emo', no_neutral=True, is_log=True, title=None, vmax=None, vmin=None):
    transition_matrix, transition_matrix_normalized = build_transition_matrix(emo_pairs, norm='row', aff=aff,
                                                                              no_neutral=no_neutral, is_log=is_log)
    # plot_matrix(transition_matrix_normalized,aff=aff,no_neutral=no_neutral,font_size=20,
    #             vmax=vmax,vmin=vmin,title=title,saveto=f'{title}.pdf')
    return transition_matrix, transition_matrix_normalized


# emotion stream analysis
def count_interval_number(series):
    return [len(each) for each in series]


def cut_list(series, cut=0):
    assert len(series) >= cut
    return series[:cut]


def return_ith(series, i=0):
    return series[i]


def get_timestep_emotions(df, n_intervals, emotion_column=None, is_counter=True):
    timestep_emotions = []
    for i in range(n_intervals):
        ith_step = df[emotion_column].apply(lambda row: return_ith(row, i)).explode().dropna()
        if is_counter:
            timestep_emotions.append(order_counter(ith_step))
        else:
            ith_step = list(ith_step)
            timestep_emotions.append(ith_step)
    return timestep_emotions


def calculate_interval_emotions(emotions, interval_size):
    # Create a DataFrame with the emotions
    df = pd.DataFrame({'emotions': emotions})

    # Assign each row an interval based on the row number and interval size
    df['interval'] = df.index // interval_size

    # Group by interval and sum emotions for each interval
    grouped = df.groupby('interval')['emotions'].sum().tolist()
    return grouped


# Define a list of emotions and set up a color palette
emotions_list = ['anger', 'disgust', 'fear', 'sadness', 'pessimism', 'surprise',
                 'neutral', 'anticipation', 'trust', 'optimism', 'joy', 'love']
palette = sns.color_palette("tab20", len(emotions_list))  # 12 distinct colors from "tab20" palette
emotion_colors = dict(zip(emotions_list, palette))  # Map each emotion to a unique color


def line_plot(emotions, n_minutes=None, xlabel=None, ylabel=None, ylim=1000, font_size=16,
              title=None, accum=False, saveto=None):
    if accum is False:
        counters = emotions
    else:
        accumulated_counter = Counter()
        counters = []
        for counter in emotions:
            accumulated_counter += counter
            counters.append(accumulated_counter.copy())

    # Convert data to DataFrame
    df = pd.DataFrame(counters).fillna(0)

    # Plotting with matplotlib
    plt.figure(figsize=(10, 6))

    # Plot using seaborn
    ax = df.plot(kind='line', marker='o', figsize=(10, 6),
                 color=[emotion_colors.get(col, 'black') for col in df.columns])

    # Add labels and title
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=font_size)
    if xlabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)
    if title is not None:
        plt.title(title, fontsize=font_size + 3)

    if n_minutes is not None:
        x_labels = [i * n_minutes for i in range(len(emotions))]
        ax.set_xticks(range(len(emotions)))
        ax.set_xticklabels(x_labels, fontsize=font_size)
    else:
        ax.set_xticks(range(len(emotions)))
        ax.set_xticklabels(list(range(len(emotions))), fontsize=font_size)

    ax.tick_params(axis='y', labelsize=font_size)
    plt.ylim(ylim)

    plt.legend(fontsize=font_size, loc='upper left', ncol=2, bbox_to_anchor=(0, 1))
    plt.grid(True)
    plt.tight_layout()

    # Compute slopes for each emotion and store in a dictionary
    x = list(range(len(df)))  # x-axis values (index positions)
    # slopes = {}
    # for col in df.columns:
    #     y = df[col]
    #     slope, intercept, r_value, p_value, std_err = linregress(x, y)
    #     slopes[col] = round(slope,2)  # Store the slope for each emotion
    #
    emotion_data = {}
    for col in df.columns:
        y = df[col].tolist()  # Convert y-axis values to a list for easier handling
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # Store the x, y, and slope for each emotion in the dictionary
        emotion_data[col] = {
            'x': x,
            'y': y,
            'slope': round(slope, 2)
        }

    if saveto is not None:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/{saveto}", bbox_inches='tight')
    plt.show()

    return emotion_data


# optional functions
def normalize_accum_emotions(emotion_dicts):
    normalized_list = []
    for emotion_counts in emotion_dicts:
        total_count = sum(emotion_counts.values())
        normalized_counts = OrderedDict(
            (emotion, round(count / total_count, 2)) for emotion, count in emotion_counts.items())
        normalized_list.append(normalized_counts)
    return normalized_list
