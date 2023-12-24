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
            filtered_hits[chain_id] = filter_template_hits(hit, query[chain_id], id_threshold)
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
    
    database_search = {'GPCRdb': query_gpcrdb, 'Empty': empty_search}

    #search_function = database_search[query['database']] if query.get('database') in database_search else empty_search
    search_function = database_search.get(query.get('database', 'Empty'))
    reference_database(query.get('database', 'Empty'))

    filtered_hits = list()
    pdbs = list()
    check_duplicates = set()

    for hit in hits:
        pdbid = hit.name[:4].upper()
        file_name = hit.name.split()[0]
        if file_name.upper() in check_duplicates:
            logging.info(f"EXCLUDED duplicate template: {file_name}")
            continue
        elif pdbid in query.get('excluded_pdb', []):
            logging.info(f"EXCLUDED template: {file_name}")
            continue
        elif pdbid not in query.get('subset_pdb', []) and query.get('subset_pdb'):
            logging.info(f"EXCLUDED template: {file_name}")
            continue
        elif not search_function(pdbid, query):
            logging.info(f"EXCLUDED template from query: {file_name}")
            check_duplicates.add(file_name.upper())
            continue
        else:
            filtered_hits.append(hit)
            logging.info(f"Selected template: {file_name}")
            check_duplicates.add(file_name.upper())
    return filtered_hits

def reference_database(db):
    if db == 'GPCRdb':
        logging.info('By using this tool please use the references in: https://gpcrdb.org/cite_gpcrdb')

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

def empty_search(pdbid, query):
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

def query_gpcrdb(pdbid, query):
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

    url = f"http://gpcrdb.org/services/structure/{pdbid}/"
    r = requests.get( url )
    rj = r.json()

    special = ['signalling_protein',
                'apo',
                'publication_date',
                'resolution',
                'ligand_function',
                'excluded_protein']

    if len(rj) != 0:
        #pdb.set_trace()
        for key, item in query.items():
            if item is None:
                continue
            if key in special:
                if key == 'signalling_protein':
                    if key in rj:
                        if rj["signalling_protein"]["type"] not in item:
                            return False
                    else:
                        return False
                elif key == 'apo':
                    if len(rj['ligands']) != 0 and item:
                        return False
                elif key == 'publication_date':
                    #pdb.set_trace()
                    date = dt.strptime(item, "%Y-%m-%d")
                    date_pdb = dt.strptime(rj[key], "%Y-%m-%d")
                    if date < date_pdb:
                        return False
                elif key == 'resolution':
                    if rj['resolution'] > item:
                        return False
                elif key == 'ligand_function':
                    for lig in rj['ligands']:
                        if lig['function'] not in item:
                            return False
                elif key == 'excluded_protein':
                    if rj['excluded_protein'] in item:
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

    return yaml_dict