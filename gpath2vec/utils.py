import requests
from requests.exceptions import ConnectionError

def get_event_hierarchy(species='9606'):
    """
    Fetches the full event hierarchy for a given species in Reactome.

    :param species: Species name (e.g., Homo sapiens) or species taxId (e.g., 9606)
    :return: JSON list object of the full event hierarchy for the given species
    """
    headers = {
        'accept': 'application/json',
    }

    url = f'https://reactome.org/ContentService/data/eventsHierarchy/{species}'

    try:
        response = requests.get(url=url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses
    except ConnectionError as e:
        print(e)
        return None
    except requests.exceptions.HTTPError as errh:
        print ("HTTP Error:",errh)
        return None
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
        return None
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
        return None
    except requests.exceptions.RequestException as err:
        print ("Something went wrong with the request",err)
        return None

    return response.json()

hierarchy = get_event_hierarchy(species='9606')
#if hierarchy is not None:
#    print(hierarchy)


def get_json_items(json_obj, key):
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            if k == key:
                yield v
            elif isinstance(v, (dict, list)):
                yield from get_json_items(v, key)
    elif isinstance(json_obj, list):
        for item in json_obj:
            yield from get_json_items(item, key)


def pathway_parent_mappings():
    parent = [p['name'] for p in hierarchy]
    pathways = [list(set(get_json_items(p, 'stId'))) for p in hierarchy]
    [pathways[i].append(hierarchy[i]['stId']) for i in range(len(pathways))]
    pathway_mappings = {parent[i]: pathways[i] for i in range(len(parent))}
    pathway_mappings_nodes = {v: k for k, values in pathway_mappings.items() for v in values}
    return pathway_mappings_nodes


def pathway_name_mappings():
    url = "https://reactome.org/download/current/ReactomePathways.txt"

    try:
        response = requests.get(url=url)
    except ConnectionError as e:
        print(e)
        return {}

    if response.status_code == 200:
        content_list = response.text.splitlines()
        entities = [tuple(d.split("\t"))[:2] for d in content_list if '-HSA' in d]
        entities = dict(entities)
    else:
        print('Status code returned a value of %s' % response.status_code)
        entities = {}

    return entities