def get_label(condition):
    if condition == 'SFix1' or condition == 'SFix2':
        return 'SFix'
    else:
        return 'Intact'