
def convert_to_high_level_icd9(icd9_code):
    k = icd9_code[:3]
    if '001' <= k <= '139':
        return 0
    elif '140' <= k <= '239':
        return 1
    elif '240' <= k <= '279':
        return 2
    elif '280' <= k <= '289':
        return 3
    elif '290' <= k <= '319':
        return 4
    elif '320' <= k <= '389':
        return 5
    elif '390' <= k <= '459':
        return 6
    elif '460' <= k <= '519':
        return 7
    elif '520' <= k <= '579':
        return 8
    elif '580' <= k <= '629':
        return 9
    elif '630' <= k <= '679':
        return 10
    elif '680' <= k <= '709':
        return 11
    elif '710' <= k <= '739':
        return 12
    elif '740' <= k <= '759':
        return 13
    elif '760' <= k <= '779':
        return 14
    elif '780' <= k <= '799':
        return 15
    elif '800' <= k <= '999':
        return 16
    elif 'E00' <= k <= 'E99':
        return 17
    elif 'V01' <= k <= 'V99':
        return 18

