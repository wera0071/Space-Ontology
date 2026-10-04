import requests, re, os, tqdm

API = "https://ru.wikipedia.org/w/api.php"
CAT_LIST = [
    "Категория:Космонавтика",
    "Категория:Космические аппараты",
    "Категория:Ракеты-носители",
    "Категория:Ракетные двигатели",
    "Категория:Космические испытания",
]

TAG_RE = re.compile(r"<.*?>|\{\{.*?\}\}|\[http.*?\]", re.S)
SESSION = requests.Session()

def members(cat, cmtype):
    cont = ''
    while True:
        p = {"action":"query","list":"categorymembers",
             "cmtitle":cat,"cmlimit":"500",
             "cmtype":cmtype,"format":"json"}
        if cont:
            p["cmcontinue"] = cont
        data = SESSION.get(API, params=p).json()
        for m in data['query']['categorymembers']:
            yield m['title']
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break

def page_text(title):
    r = SESSION.get(API, params={
        "action":"query","prop":"extracts",
        "titles":title,"explaintext":1,"format":"json"}).json()
    page = next(iter(r["query"]["pages"].values()))
    return page.get("extract","").lower()

os.makedirs("rkt_corpus/raw", exist_ok=True)
all_pages = set()

# собираем подкатегории + страницы
for cat in CAT_LIST:
    subcats = list(members(cat, "subcat"))
    for sc in subcats + [cat]:
        all_pages.update(members(sc, "page"))

print("Статей найдено", len(all_pages))

with open("rkt_corpus/raw/wiki_rkt.txt","w") as out:
    for title in tqdm.tqdm(all_pages, desc="Dow"):
        txt = page_text(title)
        txt = re.sub(TAG_RE, " ", txt)
        out.write(txt+"\n")
