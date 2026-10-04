import pymorphy3, regex as re, collections, gzip, pickle, tqdm, os

morph = pymorphy3.MorphAnalyzer()
tok   = re.compile(r"\p{L}{3,}", re.I)
cnt   = collections.Counter()

for line in open('rkt_corpus/raw/wiki_rkt.txt'):
    for w in tok.findall(line):
        lemma = morph.parse(w)[0].normal_form
        if 'NOUN' in morph.parse(lemma)[0].tag:
            cnt[lemma] += 1


with open('rkt_terms_freq.tsv','w') as f:
    for l,c in cnt.most_common():
        f.write(f"{l}\t{c}\n")


syn = pickle.load(gzip.open('/Users/apple/Downloads/taxoenrich-master/RuWordNet/synsets.pkl.gz','rb'))
lemmas_rwn = {l for s in syn for l in s['lemmas']}
new_terms = [l for l,_ in cnt.most_common(1500) if l not in lemmas_rwn]

open('candidate_terms.txt','w').write('\n'.join(new_terms))
open('rkt_final_terms.txt','w').write('\n'.join(new_terms[:400]))
print(' кандидатов для теста:', len(new_terms[:400]))
