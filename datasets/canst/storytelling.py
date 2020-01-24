import re
import os
import pdb
import shutil
import codecs
import datetime
import glob
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import pandas as pd
ch_en_dict = {
    '人魚公主': "mermaid_princess",
    '偏食的津津': "partial_eclipse",
    '偷衣服的小朋友': "children_who_steal_clothes",
    '六角星糖果': "hexagon_candy",
    '勤勤與聰聰': "two_children",
    '媽媽講故事：彩雲天使': "angel_of_clouds",
    "小人國櫻花公主（二）": "cherry_blossom_princess_2",
    "小狗花花": "puppy_flower",
    "月亮國人魚公主": "moon_country_mermaid_princess",
    "有毒的花": "poisonous_flowers",
    "松鼠森林": "squirrel_forest",
    "機械星球": "mechanical_planet",
    "河神種子": "river_god_seed",
    "滴滴仔故事": "story_of_droplet",
    "父親節的禮物": "gift_of_fathers_day",
    "蛇蛋不見了": "missing_snake_egg",
    "開朗的心情-開心每一天": "happy_every_day"
}


def get_wav_seconds(wav_file):
    try:
        sr, waveform = wavfile.read(wav_file)
        seconds = len(waveform) / sr
    except ValueError:
        pdb.set_trace()
    return round(seconds, 4)


def write_metadata(T_df, phase, out_dir):
    """
        write dataframe to train/test/val datalist
    """
    output_path = os.path.join(
        out_dir, 'cst_audio_text_' + phase + '_filelist.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for index, trow in T_df.iterrows():
            str_line = "|".join(
                [trow['text'], trow['audio_path'], trow['spk_id'], str(trow['wav_seconds'])]) + '\n'
            f.write(str_line)


def train_val_test_split():
    """
        1. here we only care about narration part of female
        split val test and train folder
        last step for preprocess
    """
    csv_file = "/home/gyzhang/speech_database/CSTelling/new_fine_cut_audio_data_1_17/CST.csv"
    out_dir = "./"
    csv_lines = pd.read_csv(filepath_or_buffer=csv_file)
    # append spk_id now
    pdb.set_trace()
    rows = csv_lines.shape[0]
    csv_lines.insert(0, "spk_id", ['69'] * rows, True)
    # reset spk id speech_type narration 69 while speech as 70
    csv_lines.loc[(csv_lines.speech_type == 'narration'), 'spk_id'] = '69'
    csv_lines.loc[(csv_lines.speech_type == 'speech'), 'spk_id'] = '70'

    # train and test split
    train_df_, test_df_ = train_test_split(
        csv_lines, test_size=0.2, random_state=42)
    split_t_test = len(test_df_) // 2
    test_df_, val_df_ = test_df_[:split_t_test], test_df_[split_t_test:]

    # add train val test
    train_df_.insert(0, "train_val", ['train'] * train_df_.shape[0], True)
    test_df_.insert(0, "train_val", ['test'] * test_df_.shape[0], True)
    val_df_.insert(0, "train_val", ['val'] * val_df_.shape[0], True)

    set_dict = {"train": train_df_, "val": val_df_, "test": test_df_}
    for phase in set_dict.keys():
        write_metadata(set_dict[phase], phase, out_dir)


def process_raw_text():
    """
        1. tranlate english to chinese
        2. extract information then save as csv file
    """
    text_file_path = "/home/gyzhang/speech_database/CSTelling/old_fine_cut_audio_data/storytelling_out_jyutping"
    audio_dir = "/home/gyzhang/speech_database/CSTelling/old_fine_cut_audio_data/sentence"
    new_audio_dir = "/home/gyzhang/speech_database/CSTelling/new_fine_cut_audio_data_1_17"
    csv_file = os.path.join(new_audio_dir, "CST.csv")

    with open(text_file_path, 'r', encoding="utf-8") as fid:
        all_files = fid.readlines()
    all_text_list = []

    for one_file in all_files:
        file_text_list = re.split("\s+", one_file.strip())
        chinese_file_name = file_text_list[0]
        speech_type = file_text_list[1]
        speech_text = file_text_list[2:]
        chinese_prefix = re.split("\_", chinese_file_name)[0]
        english_prefix = ch_en_dict[chinese_prefix]
        new_file_name = re.sub(
            chinese_prefix, english_prefix, chinese_file_name)
        audio_path = os.path.join(
            audio_dir, speech_type, chinese_file_name + '.wav')
        new_audio_path = os.path.join(
            new_audio_dir, speech_type, new_file_name + '.wav')
        wav_seconds = get_wav_seconds(audio_path)
        shutil.copyfile(audio_path, new_audio_path)
        all_text_list.append([chinese_file_name, new_file_name, speech_type,
                              wav_seconds, new_audio_path, '-'.join(speech_text)])

    df_ = pd.DataFrame(all_text_list, columns=[
                       'chinese_name', "english_name", "speech_type", "wav_seconds", "audio_path", "text"])

    df_.to_csv(csv_file, index=None, header=True)


# process_raw_text()
train_val_test_split()
