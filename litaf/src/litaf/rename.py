def rename_remove_msa_region(name, regions, paired=False, unpaired=True):
    p = 'p' if paired else ''
    u = 'u' if unpaired else ''
    message = f'_removed_msa_region_{p}{u}_' + '_'.join([f'{idx1+1}-{idx2+1}' for idx1, idx2 in regions])
    return name+message

def rename_mutate_msa(name, pos_res, paired=False, unpaired=True):
    p = 'p' if paired else ''
    u = 'u' if unpaired else ''
    message = f'_mutated_{p}{u}_' + '_'.join([str(k) for k in pos_res.keys()])
    return name+message

def rename_remove_template_from_msa(name):
    return name+'_no_template_msa'

def rename_remove_msa_features(name):
    return name+'_no_msa_features'

def rename_remove_templates(name):
    return name+'_no_template'

def rename_shuffle_templates(name, seed):
    return name+f'_shuffled_templates_{seed}'