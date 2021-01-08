import requests
import pandas as pd
from bs4 import BeautifulSoup
import json

RANKS = ['kingdom', 'clade', 'order', 'family', 'subfamily', 'supertribe', 'tribe', 'subtribe', 'genus', 'subgenus', 'section', 'subsection', 'series', 'species', 'subspecies', 'variety']

flower_names = pd.read_csv('data/flower_names.txt', sep="\n", header=None)[0]
flower_names_wiki = pd.read_csv('data/flower_names_wiki.txt', sep="\n", header=None)[0]
num_flowers = len(flower_names)

flower_data = [{'id': i, 'name':name, 'wiki': flower_names_wiki[i], 'taxonomy': []} for i, name in enumerate(flower_names)]


for flower in flower_data:
    print('============ ' + flower['name'])
    response = requests.get('https://en.wikipedia.org/wiki/' + flower['wiki'])
    soup = BeautifulSoup(response.content, 'html.parser')
    soup.br.replace_with ('\n')
    infobox_rows = soup.find('table', {'class': 'infobox'}).find_all('tr')

    for row in infobox_rows:
        cells = row.find_all('td')
        if len(cells) == 2:
            rank = cells[0].text.strip().lower().split('\n')[0]
            if rank[-1:] == ':': rank = rank[:-1]
            if rank == '(unranked)': rank = 'clade'
            name = cells[1].text.strip().lower().split('\n')[0]

            if rank not in RANKS: print('Unkown Rank: ' + rank)
            flower['taxonomy'].append({rank: name})
            # print(rank + ': ' + name)

with open('flower_data.json', 'w') as fp:
    json.dump(flower_data, fp)
