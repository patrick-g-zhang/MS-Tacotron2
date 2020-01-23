import pickle
import pdb
import re
import pandas as pd
import pdb


class Candict(object):
    """docstring for Candict"""

    def __init__(self, file_or_path="./data/can_syl_dictionary"):
        super(Candict, self).__init__()
        self.candict = self.parser_can_dict(file_or_path)

    def write_can_dict(self, syl_dict='./syl_dict', write_path="./data/can_syl_dictionary"):
        """
                        transform a json file to text based dict
        """
        pkl_file = open(syl_dict, 'rb')
        syl_dict = pickle.load(pkl_file)
        pkl_file.close()
        wfid = open(write_path, 'w')
        for syl in syl_dict:
            sylif = syl_dict[syl]
            sylif.insert(0, syl)
            wfid.write("  ".join(sylif) + '\n')
        wfid.close()

    def parser_can_dict(self, file_or_path):
        with open(file_or_path, 'r') as rfid:
            entries = rfid.readlines()
        can_dict = {}
        for entry in entries:
            syl_if = re.split("\s+", entry.strip())
            syl = syl_if[0]
            intialf = syl_if[1:]
            can_dict[syl] = intialf
        return can_dict

    def load_if(self, write_path="./data/can_phone"):
        infi_set = []
        pkl_file = open('./final_set', 'rb')
        final_set = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open('./initial_set', 'rb')
        initial_set = pickle.load(pkl_file)
        pkl_file.close()
        infi_set.extend(initial_set)
        infi_set.extend(final_set)
        infi_df = pd.DataFrame(infi_set)
        infi_df.to_csv(write_path, sep="\t")

    def get_diff(self, write_path="./data/can_phone"):
        all_phone_list = []
        for syl in self.candict:
            all_phone_list.extend(self.candict[syl])
        all_phone_list = list(set(all_phone_list))
        infi_df = pd.DataFrame(all_phone_list)
        infi_df.to_csv(write_path, sep="\t", header=False, index=False)


# cd = Candict()
# cd.get_diff()
