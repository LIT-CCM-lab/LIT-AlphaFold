'''
Functions for template filtering

* *filter_templates*, filter templates for multichain input
* *filter_template_hits*, filter templates for monomer input
* *generate_query*, write results in text form
* *empty_search*, search function returning always True
* *query_gpcrdb*, query GPCRdb
* *load_template_query*, read query file
'''
import logging
from datetime import datetime as dt

import requests
import yaml

def filter_templates(hits, query):
    '''
    Filter template hits based on a user defined query

    Parameters
    ----------
    hits: str
        Template hits generated ater a search
    query: dict
        Filters on the templates

    Return
    ------
    filtered structures: str

    '''
    filtered_hits = {}
    for chain_id, hit in hits.items():
        logging.info(f"Filetring templates for {chain_id}")
        if chain_id in query:
            filtered_hits[chain_id] = filter_template_hits(hit, query[chain_id])
        else:
            filtered_hits[chain_id] = hit

    return filtered_hits


def filter_template_hits(hits, query):
    '''
    General function for evaluation of template hits
    To query an annotated structural database it is necessary to generate a function taking as arguments the pdbid and the dict query 
    and returning True if the PDBID matches the criteris and False if not

    Parameters
    ----------
    hits: str
        Template hits generated ater a search
    query: dict
        Filters on the templates

    Return
    ------
    filtered structures: str
    '''

    return template_filter([hit.name for hit in hits], query)
    

def filter_template_features(feature_dict, query):
    filtered_templates = template_filter(feature_dict['template_domain_names'], query)
    idxs = [i for i, pdbid in enumerate(feature_dict['template_domain_names']) if pdbid in filtered_templates]
    feature_dict["template_all_atom_positions"] = feature_dict["template_all_atom_positions"][idxs]
    feature_dict["template_all_atom_masks"] = feature_dict["template_all_atom_masks"][idxs]
    feature_dict["template_sequence"] = feature_dict["template_sequence"][idxs]
    feature_dict["template_aatype"] = feature_dict["template_aatype"][idxs]
    feature_dict["template_domain_names"] = feature_dict["template_domain_names"][idxs]
    feature_dict["template_sum_probs"] = feature_dict["template_sum_probs"][idxs]

    return feature_dict


def template_filter(pdbids_chain, query):
    database_search = {'GPCRdb_r': query_gpcrdb_r, 'GPCRdb_g': query_gpcrdb_g, 'Empty': empty_search}

    #search_function = database_search[query['database']] if query.get('database') in database_search else empty_search
    query, all_entries = query
    search_function = database_search.get(query.get('database', 'Empty'))
    
    filtered_hits = list()
    excluded_hits = list()
    excluded_query_hits = list()
    selected_hits = list()
    check_duplicates = set()

    for pdbid_chain in pdbids_chain:
        pdbid = pdbid_chain[:4].upper()
        file_name = pdbid_chain.upper()
        if isinstance(pdbid_chain, bytes):
            pdbid = pdbid.decode('utf-8')
            file_name = pdbid.decode('utf-8')
        if file_name.upper() in check_duplicates:
            continue
        elif pdbid in query.get('excluded_pdb', []):
            excluded_hits.append(file_name)
            continue
        elif pdbid not in query.get('subset_pdb', []) and query.get('subset_pdb'):
            excluded_hits.append(file_name)
            continue
        elif not search_function(file_name, query, all_entries):
            excluded_query_hits.append(file_name)
            check_duplicates.add(file_name)
            continue
        else:
            filtered_hits.append(pdbid_chain)
            selected_hits.append(file_name)
            check_duplicates.add(file_name)
    if len(excluded_hits) > 0:
        logging.info(f"EXCLUDED templates: {' '.join(excluded_hits)}")
    if len(excluded_query_hits) > 0:
        logging.info(f"EXCLUDED templates: {' '.join(excluded_query_hits)}")
    if len(excluded_hits) > 0 or len(excluded_query_hits) > 0:
        logging.info(f"Selected templates: {' '.join(selected_hits)}")
    return filtered_hits


def get_all_entries(db):
    if db in ['GPCRdb_r', 'GPCRdb_g']:
        return get_all_gpcrdb()
    else:
        return None


def get_all_gpcrdb():
    logging.info("Downloading GPCRdb data, please cite this work as indicated on: https://gpcrdb.org/cite_gpcrdb")
    url = f"http://gpcrdb.org/services/structure/"
    r = requests.get( url )
    rj = r.json()
    return {r["pdb_code"]: r for r in rj}


def generate_filter(hits):
    '''
    Generate query finding the same results as the given hit list

    Parameters
    ----------
    hits: str
        Template hits generated ater a search

    Return
    ------
    query containing the filtered hits: dict
    '''
    accepted_pdbs = []
    for hit in hits:
        accepted_pdbs.append(hit.name[:4].upper())
    return {'subset_pdb': accepted_pdbs}


def empty_search(*args):
    '''
    Placeholder function always returning True regardless of the passed inputs

    Parameters
    ----------
    pdbid: str
        PDBID of the structure
    query: dict
        Criteria for the selected structure

    Return
    ------
    The PDBID structure matches the query requirements: bool
    '''
    return True


def query_gpcrdb_r(pdbid_chain, query, all_entries):
    '''
    Compare the entry for a specific PDBID on GPCRdb to the user defined parameters

    The comparison is based on identity between the user selection.
    Special attention is given to publication_date which removes all structures published after the given date the dae is in the YYYY-MM-DD format
    resoultion is used to set a upper limit for the resoultion value in angstrom
    The function returns True if the entry satisfies the results or False if a criteria is unmet or if the structures is not in GPCRdb

    Parameters
    ----------
    pdbid: str
        PDBID of the structure
    query: dict
        Criteria for the selected structure 

    Return
    ------
    The PDBID structure matches the query requirements: bool
    '''

    pdbid, chainid = pdbid_chain.split('_')

    rj = all_entries.get(pdbid, {})

    if len(rj) != 0:
        return compare_gpcrdb_r_entry(pdbid, chainid, query, rj)
    else:
        logging.info(f'PDBID: {pdbid} not in GPCRdb')
        return False


def compare_gpcrdb_r_entry(pdbid, chainid, query, entry):
    special = ['signalling_protein',
                'apo',
                'publication_date',
                'resolution',
                'ligand_function',
                'excluded_protein']

    if entry["preferred_chain"] != chainid:
        return False
    for key, item in query.items():
        if item is None:
            continue
        if key in special:
            if key == 'signalling_protein':
                if key in entry:
                    if entry["signalling_protein"]["type"] not in item:
                        return False
                else:
                    return False
            elif key == 'apo':
                if len(entry['ligands']) != 0 and item:
                    return False
            elif key == 'publication_date':
                #pdb.set_trace()
                date = dt.strptime(item, "%Y-%m-%d")
                date_pdb = dt.strptime(entry[key], "%Y-%m-%d")
                if date < date_pdb:
                    return False
            elif key == 'resolution':
                if entry['resolution'] > item:
                    return False
            elif key == 'ligand_function':
                for lig in entry['ligands']:
                    if lig['function'] not in item:
                        return False
            elif key == 'excluded_protein':
                if entry['protein'] in item:
                    return False
        elif key in entry:
            if entry[key] not in item:
                return False
    return True


def query_gpcrdb_g(pdbid_chain, query):
    '''
    Compare the entry for a specific PDBID on GPCRdb/GProteindb to the user defined parameters
    This tool focuses on G-protein, however only complexes bound to a receptor can currently be queried

    The comparison is based on identity between the user selection.
    Special attention is given to publication_date which removes all structures published after the given date the dae is in the YYYY-MM-DD format
    resoultion is used to set a upper limit for the resoultion value in angstrom
    The function returns True if the entry satisfies the results or False if a criteria is unmet or if the structures is not in GPCRdb

    Parameters
    ----------
    pdbid: str
        PDBID of the structure
    query: dict
        Criteria for the selected structure 

    Return
    ------
    The PDBID structure matches the query requirements: bool
    '''

    pdbid, chainid = pdbid_chain.split('_')

    rj = get_gpcrdb_entry(pdbid)

    special = ['publication_date',
                'resolution',
                'excluded_protein',
                'species']

    if len(rj) != 0:
        if rj.get("signalling_protein", None) is None:
            return False
        species = None
        for e_values in rj["signalling_protein"]["data"].values:
            if e_values["chain"] == chainid:
                species = e_values["entry_name"].split('_')[1]
        if species is None:
            return False
        for key, item in query.items():
            if item is None:
                continue
            if key in special:
                if key == 'publication_date':
                    #pdb.set_trace()
                    date = dt.strptime(item, "%Y-%m-%d")
                    date_pdb = dt.strptime(rj[key], "%Y-%m-%d")
                    if date < date_pdb:
                        return False
                elif key == 'resolution':
                    if rj['resolution'] > item:
                        return False
                elif key == 'excluded_protein':
                    if rj['excluded_protein'] in item:
                        return False
                elif key == 'species':
                    if species != item:
                        return False
            elif key in rj:
                if rj[key] not in item:
                    return False
        return True
    else:
        logging.info(f'PDBID: {pdbid} not in GPCRdb')
        return False


def load_template_filter(file):
    '''
    Read query from .yaml file

    Parameters
    ----------
    file: str
        .yaml file containing the query

    Return
    ------
    Structure annotation query: dict
    '''
    logging.info(f'Reading template information from {file}')
    with open(file, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.loader.SafeLoader)

    all_entries = get_all_entries(yaml_dict.get('database', 'Empty'))

    return [yaml_dict, all_entries]