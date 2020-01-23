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
    return round(seconds,4)


def pattern_check(file_text, file_name):
    """
        check file_text excepted pattern
        keoi5-dei2-jat1gaa1jan4-hai2-luk6-dei6
        return: keoi5-dei2-jat1-gaa1-jan4-hai2-luk6-dei6
    """
    syls = re.split("\-", file_text)
    all_syls = []
    for syl in syls:
        ss = re.search(r"(\d)", syl[:-1])
        if ss is not None:
            new_syls = re.split(r"(\d)", syl)
            assert new_syls[-1] == ""
            num_syls = len(new_syls[:-1]) // 2
            for num in range(num_syls):
                assert new_syls[2 * num + 1].isdigit()
                new_syl = new_syls[2 * num] + new_syls[2 * num + 1]
                all_syls.append(new_syl)
        else:
            all_syls.append(syl)
    return '-'.join(all_syls)


def write_metadata(T_list, phase, out_dir):
    output_path = os.path.join(
        out_dir, 'cst_audio_text_' + phase + '_filelist.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in T_list:
            f.write(line)


def dur_infos():
    fid = open("./nnew_storytelling_text.txt", 'w', encoding='utf-8')
    with open("./new_storytelling_text.txt", 'r', encoding='utf-8') as sid:
        file_paths = sid.readlines()
    for filepath in file_paths:
        text, wav_path, spk_id = re.split("\|", filepath.strip())
        fid.write("{0}|{1}|{2}|{3}\n".format(text, wav_path,
                                             spk_id, str(get_wav_seconds(wav_path))))
    fid.close()


def train_val_test_split():
    """
        1. here we only care about narration part of female
        split val test and train folder
        last step for preprocess
    """
    csv_file = "/home/gyzhang/speech_database/CSTelling/new_fine_cut_audio_data_1_17/CST.csv"
    csv_lines = pd.read_csv(filepath_or_buffer=csv_file)
    csv_lines.query("speech_type == 'narration'")
    # with open("./new_storytelling_text.txt", 'r', encoding='utf-8') as sid:
        # file_paths = sid.readlines()
    # append spk_id now 
    rows = csv_lines.shape[0]
    pdb.set_trace()
    csv_lines.insert(0, "spk_id", ['69']*rows, True)
    # csv_lines[[ 'wav_seconds']].apply(lambda x:str(x), axis=1)
    csv_selected_lines = csv_lines[['text', 'audio_path', 'spk_id', 'wav_seconds']]
    str_lines = csv_selected_lines.apply(lambda x: "|".join([xi if isinstance(xi, str) else str(xi)  for xi in x ]) + "\n", axis=1)
    str_lines = str_lines.tolist()
    train_set, t_test = train_test_split(
        str_lines, test_size=0.2, random_state=42)
    split_t_test = len(t_test) // 2
    valid_set = t_test[0:split_t_test]
    test_set = t_test[split_t_test:]
    set_dict = {"train": train_set, "val": valid_set, "test": test_set}
    for phase in set_dict.keys():
        write_metadata(set_dict[phase], phase, "./")

def ch_en_trans(src_dir, target_dir):
    """
        translate all chinese file name to english name
        媽媽講故事：彩雲天使_seg35 -> angel_of_clouds
        Args:
            src_dir:source file dir
            targte_dir: target file dir 
    """
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for wav_path in glob.glob(src_dir + "/*.wav"):
        file_name = os.path.basename(wav_path)
        cn_name = re.split("\_", file_name)[0]
        new_file_name = re.sub(cn_name, ch_en_dict[cn_name], file_name)
        full_path_new_file_name = os.path.join(
            target_dir, new_file_name)
        shutil.copyfile(wav_path, full_path_new_file_name)

# 1. translate chinese name to english name
# ch_en_trans("/home/gyzhang/speech_database/CSTelling/prepared_story_telling_1_6", "/home/gyzhang/speech_database/CSTelling/new_prepared_story_telling")

def process_raw_st(prepared_dir, prepared_txt, new_st_dir, spk_id):
    """
        preprocess_raw story telling data: 
        1. translate chinese character to english
        Args:
            prepared_dir: original data dir
            prepared_txt: txt file with jupyting
            new_st_dir: new data dir
    """
    if not os.path.exists(new_st_dir):
        os.mkdir(new_st_dir)

    full_file_txt = os.path.join(prepared_dir, prepared_txt)
    new_full_file_txt = os.path.join(new_st_dir, prepared_txt)
    story_info_path = os.path.join(new_st_dir, 'story_info.txt')
    with open(full_file_txt, 'r', encoding="utf-8") as fid:
        all_files = fid.readlines()
    new_fid = open(new_all_file_txt, 'w')
    si_fid = open(story_info_path, 'w')

    pre_s_name = ''
    total_dur = 0
    for s_num, stfile in enumerate(all_files):
        file_name, file_text = re.split('\s+', stfile.strip())
        full_path_name = os.path.join(prepared_dir, file_name + '.wav')
        assert os.path.exists(full_path_name)

        # pdb.set_trace()
        new_file_text = pattern_check(file_text, file_name)
        s_name = re.split('\_', file_name)[0]

        # start a new story, reset duration
        if s_name != pre_s_name or s_num == len(all_files) - 1:
            if s_num != 0:
                # if s_num == len(all_files) - 1:
                    # total_dur += get_wav_seconds(full_path_name)

                story_infos = "{0} | {1} \n".format(
                    pre_s_name, ch_en_dict[pre_s_name])  # , str(datetime.timedelta(seconds=int(total_dur))))
                si_fid.write(story_infos)
                total_dur = 0

        # new wav file
        new_file_name = re.sub(s_name, ch_en_dict[s_name], file_name)
        full_path_new_file_name = os.path.join(
            new_st_dir, new_file_name + '.wav')

        # seconds = get_wav_seconds(full_path_name)
        # total_dur += seconds
        shutil.copyfile(full_path_name, full_path_new_file_name)
        output_infos = "{0}|{1}|{2}\n".format(
            full_path_new_file_name, new_file_text, str(spk_id))
        new_fid.write(output_infos)

        pre_s_name = s_name
    new_fid.close()
    si_fid.close()


def process_raw_text():
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
        new_file_name = re.sub(chinese_prefix, english_prefix, chinese_file_name)
        audio_path = os.path.join(audio_dir, speech_type, chinese_file_name + '.wav')
        new_audio_path = os.path.join(new_audio_dir, speech_type, new_file_name + '.wav')
        wav_seconds = get_wav_seconds(audio_path)
        shutil.copyfile(audio_path, new_audio_path)
        all_text_list.append([chinese_file_name, new_file_name, speech_type, wav_seconds, new_audio_path,  '-'.join(speech_text)])
    
    df_ = pd.DataFrame(all_text_list, columns=['chinese_name', "english_name", "speech_type", "wav_seconds", "audio_path", "text"])
    df_.to_csv(csv_file, index=None, header=True)

# process_raw_text()
train_val_test_split()