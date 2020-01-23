import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import cusent
# from hparams import train_hparams as hparams
import glob
import pdb
import re
from shutil import copyfile


def preprocess_cusent(args):
    in_dir = os.path.join(args.base_dir, 'CUSENT_wav/newtrain')
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    all_T_train, all_T_valid, all_T_test, spk_id_dict = cusent.build_from_path(
        in_dir, args.num_workers, tqdm=tqdm)
    T_dict = {'train': all_T_train, 'val': all_T_valid, 'test': all_T_test}
    pdb.set_trace()
    for phase in ['train', 'val', 'test']:
        write_metadata(T_dict[phase], phase, out_dir)


def write_metadata(T_list, phase, out_dir):
    output_path = os.path.join(
        out_dir, 'cusent_audio_text_' + phase + '_filelist.txt')
    total_seconds = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in T_list:
            total_seconds = total_seconds + line[-1]
            f.write('|'.join([str(x) for x in line]) + '\n')
    hours = total_seconds // 3600
    minutes = (total_seconds - hours * 3600) // 60
    print("{0}:{1}h{2}mins".format(phase, str(hours), str(minutes)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir', default="/home/gyzhang/speech_database/CUSENT")
    parser.add_argument(
        '--output', default='/home/gyzhang/projects/tacotron2/datasets/cusent-female-filelists')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--train_test_split', type=bool, default=True)
    parser.add_argument('--process_all', type=bool, default=False)
    args = parser.parse_args()
    preprocess_cusent(args)


if __name__ == "__main__":
    main()
