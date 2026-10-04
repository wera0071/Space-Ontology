import re, pymorphy3, tqdm

KEYWORDS = ('ракет', 'двигател', 'спутник', 'орбит',
            'турбо', 'насос', 'ступен', 'разгонн',
            'жрд', 'лун', 'марс', 'блок', 'носител',
            'апогей', 'перигей', 'байконур', 'экспедици')

STOP = set('год земля человек время работа результат программа проект сайт примечание ссылка'.split())

morph = pymorphy3.MorphAnalyzer()
out = open('rkt_final_terms.txt','w')
for line in open('candidate_terms.txt'):
    w = line.strip()
    if len(w) < 5:                 
        continue
    if w in STOP:                  
        continue
    if any(k in w for k in KEYWORDS):
        out.write(w + '\n')
out.close()
print('готов')
