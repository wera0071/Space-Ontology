import gzip, pickle, xml.etree.ElementTree as ET, tqdm, os, glob, re

RWN_DIR = 'RuWordNet'          
OUT     = 'RuWordNet/synsets.pkl.gz'

def parse_synsets(path):
    for event, elem in ET.iterparse(path):
        if elem.tag == 'synset':
            lemmas = [l.text for l in elem.findall('./lemma')]
            yield {'id': elem.get('id'), 'lemmas': lemmas}
            elem.clear()

all_synsets = []
for xml_path in glob.glob(os.path.join(RWN_DIR, '*synsets*.xml')):
    for syn in tqdm.tqdm(parse_synsets(xml_path), desc=os.path.basename(xml_path)):
        all_synsets.append(syn)

with gzip.open(OUT, 'wb') as f:
    pickle.dump(all_synsets, f)

print("сохранено:", OUT, "| всего synsets:", len(all_synsets))
