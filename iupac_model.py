import re

def smiles_atom_tokenizer (smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def split_tokens(tokens_to_be_splitted, all_tokens):
    if ''.join(tokens_to_be_splitted) in all_tokens:
        indexes = duplicates(all_tokens, ''.join(tokens_to_be_splitted))
        for k, idx in enumerate(indexes):
            all_tokens[idx] = tokens_to_be_splitted[0]
            for i in range(1, len(tokens_to_be_splitted)):
                all_tokens.insert(idx+i, tokens_to_be_splitted[i])
            if k < len(indexes) - 1:
                indexes[k+1:] = list(map(lambda x: x+(len(tokens_to_be_splitted) - 1), indexes[k+1:]))
def add_symbol(pat):
    new_patt = pat[1:].split(',')
    ls_del = []
    if new_patt[0][0] == '0':
        ls_del.append(new_patt[0][0])
        new_patt[0] = new_patt[0][1:]

    while int(new_patt[0]) > int(new_patt[1]):
        ls_del.append(new_patt[0][0])
        new_patt[0] = new_patt[0][1:]
    result = ['.', ''.join(ls_del), '^', new_patt[0], ',', new_patt[1]]
    return result

def check_correct_tokenize(iupac,tokens):
    if ''.join(tokens).replace('^', '') == iupac:
        return True
    else:
        return False

def iupac_tokenizer(iupac):

    pattern = "\.\d+,\d+|[1-9]{2}\-[a-z]\]|[0-9]\-[a-z]\]|[1-9]{2}[a-z]|[1-9]{2}'[a-z]|[0-9]'[a-z]|[0-9][a-z]|\([0-9]\+\)|\([0-9]\-\)|" + \
              "[1-9]{2}|[0-9]|-|\s|\(|\)|S|R|E|Z|N|C|O|'|\"|;|λ|H|,|\.|\[[a-z]{2}\]|\[[a-z]\]|\[|\]|"

    alcane = "methane|methanoyl|methan|ethane|ethanoyl|ethan|propanoyl|propane|propan|propa|butane|butanoyl|butan|buta|pentane|" + \
             "pentanoyl|pentan|hexane|hexanoyl|hexan|heptane|heptanoyl|heptan|octane|octanoyl|octan|nonane|nonanoyl|" + \
             "nonan|decane|decanoyl|decan|icosane|icosan|cosane|cosan|contane|contan|"

    pristavka_name = "hydroxide|hydroxyl|hydroxy|hydrate|hydro|cyclo|spiro|iso|"
    pristavka_digit = "mono|un|bis|bi|dicta|di|tetraza|tetraz|tetra|tetr|pentaza|pentaz|penta|hexaza|" + \
                      "hexa|heptaza|hepta|octaza|octa|nonaza|nona|decaza|deca|kis|"

    prefix_alcane = "methylidene|methyl|ethyl|isopropyl|propyl|isobutyl|sec-butyl|tert-butyl|butyl|pentyl|hexyl|heptyl|octyl|"
    carbon = "meth|eth|prop|but|pent|hex|hept|oct|non|dec|icosa|icos|cosa|cos|icon|conta|cont|con|heni|hene|hen|hecta|hect|"

    prefix_all = "benzhydryl|benzoxaza|benzoxaz|benzoxa|benzox|benzo|benzyl|benz|phenacyl|phenanthro|phenyl|phenoxaza|phenoxaz|phenoxy|phenox|phenol|pheno|phen|acetyl|aceto|acet|" + \
                 "peroxy|oxido|oxino|oxalo|oxolo|oxocyclo|oxol|oxoc|oxon|oxo|oxy|pyrido|pyrimido|imidazo|naphtho|stiboryl|stibolo|"

    prefix_isotope = "protio|deuterio|tritio|"
    suffix_isotope = "protide|"
    prefix_galogen = "fluoro|fluoranyl|fluoridoyl|fluorido|chloro|chloranyl|chloridoyl|chlorido|bromo|bromanyl|bromidoyl|bromido|iodo|iodanyl|iodidoyl|iodanuidyl|iodido|"
    suffix_galogen = "fluoride|chloride|chloridic|perchloric|bromide|iodide|iodane|hypoiodous|hypochlorous|"
    prefix_сhalcogen = "phosphonato|phosphoroso|phosphonia|phosphoryl|phosphanyl|arsono|arsanyl|stiba|"
    suffix_сhalcogen = "phosphanium|phosphate|phosphite|phosphane|phosphanide|phosphonamidic|phosphonous|phosphinous|phosphinite|phosphono|arsonic|stibane|"
    prefix_metal = "alumanyl|gallanyl|stannyl|plumbyl|"
    suffix_metal = "chromium|stannane|gallane|alumane|aluminane|aluminan|"
    prefix_non_metal = "tellanyl|germanyl|germyl|"
    suffix_non_metal = "germane|germa|"

    prefix_sulfur = "sulfanylidene|sulfinamoyl|sulfonimidoyl|sulfinimidoyl|sulfamoyl|sulfonyl|sulfanyl|sulfinyl|sulfinato|sulfenato|" + \
                    "sulfonato|sulfonio|sulfino|sulfono|sulfido|"
    suffix_sulfur = "sulfonamide|sulfinamide|sulfonamido|sulfonic|sulfamic|sulfinic|sulfuric|thial|thione|thiol|" + \
                    "sulfonate|sulfite|sulfate|sulfide|sulfinate|sulfanium|sulfamate|sulfane|sulfo|"

    prefix_nitrogen = "hydrazono|hydrazino|nitroso|nitrous|nitro|formamido|amino|amido|imino|imido|anilino|anilin|thiocyanato|cyanato|cyano|azido|azanidyl|azanyl|" + \
                      "azanide|azanida|azonia|azonio|amidino|nitramido|diazo|"
    suffix_nitrogen = "ammonium|hydrazide|hydrazine|hydrazin|amine|imine|oxamide|nitramide|formamide|cyanamide|amide|imide|amidine|isocyanide|azanium|" + \
                      "thiocyanate|cyanate|cyanic|cyanatidoyl|cyanide|nitrile|nitrite|hydrazonate|"

    suffix_carbon = "carbonitrile|carboxamide|carbamimidothioate|carbodithioate|carbohydrazonate|carbonimidoyl|carboximidoyl|" + \
                    "carbamimidoyl|carbamimidate|carbamimid|carbaldehyde|carbamate|carbothioyl|carboximidothioate|carbonate|" + \
                    "carboximidamide|carboximidate|carbamic|carbonochloridate|carbothialdehyde|carbothioate|carbothioic|carbono|carbon|carbo|" + \
                    "formate|formic|"
    prefix_carbon = "carboxylate|carboxylato|carboxylic|carboxy|halocarbonyl|carbamoyl|carbonyl|carbamo|thioformyl|formyl|"

    silicon = "silanide|silane|silole|silanyl|silyloxy|silylo|silyl|sila|"
    boron = "boranyl|boranuide|boronamidic|boranuida|boranide|borinic|borate|borane|boran|borono|boron|bora|"
    selenium = "selanyl|seleno|"

    suffix_all = "ane|ano|an|ene|enoxy|eno|en|yne|yn|yl|peroxol|peroxo|" + \
                 "terephthalate|terephthalic|phthalic|phthalate|oxide|oate|ol|oic|ic|al|ate|ium|one|"

    carbon_trivial = "naphthalen|naphthal|inden|adamant|fluoren|thiourea|urea|anthracen|acenaphthylen|" + \
                     "carbohydrazide|annulen|aniline|acetaldehyde|benzaldehyde|formaldehyde|phthalaldehyde|acephenanthrylen|" + \
                     "phenanthren|chrysen|carbanid|chloroform|fulleren|cumen|formonitril|fluoranthen|terephthalaldehyde|azulen|picen|" + \
                     "pyren|pleiaden|coronen|tetracen|pentacen|perylen|pentalen|heptalen|cuban|hexacen|oxanthren|ovalen|aceanthrylen|"

    heterocycles = "indolizin|arsindol|indol|furan|furo|piperazin|pyrrolidin|pyrrolizin|thiophen|thiolo|imidazolidin|imidazol|pyrimidin|pyridin|" + \
                    "piperidin|morpholin|pyrazol|pyridazin|oxocinnolin|cinnolin|pyrrol|thiochromen|oxochromen|chromen|quinazolin|phthalazin|quinoxalin|carbazol|xanthen|pyrazin|purin|" + \
                    "indazol|naphthyridin|quinolizin|guanidin|pyranthren|pyran|thianthren|thian|acridin|acrido|yohimban|porphyrin|pteridin|tetramin|pentamin|" + \
                    "borinin|borino|boriran|borolan|borol|borinan|phenanthridin|quinolin|perimidin|corrin|phenanthrolin|phosphinolin|indacen|silonin|borepin|"

    prefix_heterocycles = "thiaz|oxaza|oxaz|oxan|oxa|ox|aza|az|thia|thioc|thion|thio|thi|telluro|phospha|phosph|selen|bor|sil|alum|ars|germ|tellur|imid|"

    suffix_heterocycles = "ir|et|olo|ol|ino|in|ep|oc|on|ec|"
    saturated_unsatured = "idine|idene|idin|ane|an|ine|in|id|e|"
    pristavka_exception = "do|trisodium|tris|triacetyl|triamine|triaza|triaz|tria|trityl|tri|o"

    type_ = "acid|ether|"
    element = "hydrogen|helium|lithium|beryllium|nitrogen|oxygen|fluorine|neon|sodium|magnesium|aluminum|silicon|" + \
              "phosphorus|sulfur|chlorine|argon|potassium|calcium|scandium|titanium|vanadium|chromium|manganese|iron|" + \
              "cobalt|nickel|copper|zinc|gallium|germanium|arsenic|selenium|bromine|krypton|rubidium|yttrium|zirconium|" + \
              "niobium|molybdenum|technetium|ruthenium|rhodium|palladium|silver|cadmium|indium|antimony|tellurium|iodine|" + \
              "xenon|cesium|barium|lanthanum|cerium|praseodymium|neodymium|latinum|promethium|samarium|europium|gadolinium|" + \
              "terbium|dysprosium|holmium|erbium|thulium|ytterbium|lutetium|hafnium|tantalum|tungsten|rhenium|osmium|" + \
              "iridium|platinum|gold|aurum|mercury|thallium|lead|bismuth|polonium|astatine|radon|francium|radium|actinium|" + \
              "thorium|protactinium|uranium|neptunium|plutonium|americium|curium|berkelium|einsteinium|fermium|californium|" + \
              "mendelevium|nobelium|lawrencium|rutherfordium|dubnium|seaborgium|bohrium|hassium|meitnerium|tin|"

    other_ions = "perchlorate|perbromate|periodate|hypofluorite|hypochlorite|hypobromite|hypoiodite|nitrate|silicate|hydride|"

    regex = re.compile(pattern + heterocycles + carbon_trivial + type_ + element + prefix_isotope + other_ions + alcane + pristavka_digit + pristavka_name + prefix_alcane + \
                       carbon + silicon + prefix_nitrogen + prefix_sulfur + prefix_carbon + prefix_metal + prefix_non_metal + prefix_all + prefix_galogen + prefix_сhalcogen + \
                       suffix_carbon + suffix_nitrogen + suffix_sulfur + suffix_galogen + suffix_сhalcogen + suffix_metal + suffix_non_metal + suffix_all + suffix_heterocycles + \
                       suffix_isotope + boron + selenium  + prefix_heterocycles + saturated_unsatured + pristavka_exception)
    tokens = [token for token in regex.findall(iupac)]

    split_tokens(['meth', 'ane'], tokens)
    split_tokens(['meth', 'an'], tokens)
    split_tokens(['eth', 'ane'], tokens)
    split_tokens(['eth', 'an'], tokens)
    split_tokens(['prop', 'ane'], tokens)
    split_tokens(['prop', 'an'], tokens)
    split_tokens(['but', 'ane'], tokens)
    split_tokens(['but', 'an'], tokens)
    split_tokens(['pent', 'ane'], tokens)
    split_tokens(['pent', 'an'], tokens)
    split_tokens(['hex', 'ane'], tokens)
    split_tokens(['hex', 'an'], tokens)
    split_tokens(['hept', 'ane'], tokens)
    split_tokens(['hept', 'an'], tokens)
    split_tokens(['oct', 'ane'], tokens)
    split_tokens(['oct', 'an'], tokens)
    split_tokens(['non', 'ane'], tokens)
    split_tokens(['non', 'an'], tokens)
    split_tokens(['dec', 'ane'], tokens)
    split_tokens(['dec', 'an'], tokens)
    split_tokens(['cos', 'ane'], tokens)
    split_tokens(['cos', 'an'], tokens)
    split_tokens(['cont', 'ane'], tokens)
    split_tokens(['cont', 'an'], tokens)
    split_tokens(['icos', 'ane'], tokens)
    split_tokens(['icos', 'an'], tokens)

    split_tokens(['thi', 'az'], tokens)
    split_tokens(['thi', 'oc'], tokens)
    split_tokens(['thi', 'on'], tokens)
    split_tokens(['benz', 'ox'], tokens)
    split_tokens(['benz', 'oxa'], tokens)
    split_tokens(['benz', 'ox', 'az'], tokens)
    split_tokens(['benz', 'ox', 'aza'], tokens)
    split_tokens(['phen', 'ox'], tokens)
    split_tokens(['phen', 'oxy'], tokens)
    split_tokens(['phen', 'oxa'], tokens)
    split_tokens(['phen', 'ox', 'az'], tokens)
    split_tokens(['phen', 'ox', 'aza'], tokens)
    split_tokens(['phen', 'ol'], tokens)
    split_tokens(['en', 'oxy'], tokens)
    split_tokens(['ox', 'az'], tokens)
    split_tokens(['ox', 'aza'], tokens)
    split_tokens(['tri', 'az'], tokens)
    split_tokens(['tri', 'amine'], tokens)
    split_tokens(['tri', 'acetyl'], tokens)
    split_tokens(['ox', 'ol'], tokens)
    split_tokens(['ox', 'olo'], tokens)
    split_tokens(['ox', 'an'], tokens)
    split_tokens(['ox', 'oc'], tokens)
    split_tokens(['ox', 'on'], tokens)
    split_tokens(['tri', 'az'], tokens)
    split_tokens(['tri', 'aza'], tokens)
    split_tokens(['tri', 'sodium'], tokens)
    split_tokens(['tetr', 'az'], tokens)
    split_tokens(['tetr', 'aza'], tokens)
    split_tokens(['pent', 'az'], tokens)
    split_tokens(['pent', 'aza'], tokens)
    split_tokens(['hex', 'aza'], tokens)
    split_tokens(['hept', 'aza'], tokens)
    split_tokens(['oct', 'aza'], tokens)
    split_tokens(['non', 'aza'], tokens)
    split_tokens(['dec', 'aza'], tokens)
    split_tokens(['oxo', 'chromen'], tokens)
    split_tokens(['oxo', 'cinnolin'], tokens)
    split_tokens(['oxo', 'cyclo'], tokens)
    split_tokens(['thio', 'chromen'], tokens)
    split_tokens(['thio', 'cyanato'], tokens)

    if (len(re.findall(re.compile('[0-9]{2}\-[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('[0-9]{2}\-[a-z]\]'), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])
                tokens.insert(i+2,tok[3])
                tokens.insert(i+3,tok[4])

    if (len(re.findall(re.compile('[0-9]\-[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('[0-9]\-[a-z]\]'), tok):
                tokens[i] = tok[:1]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])
                tokens.insert(i+3,tok[3])

    if (len(re.findall(re.compile('\[[a-z]{2}\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('\[[a-z]{2}\]'), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])
                tokens.insert(i+3,tok[3])

    if (len(re.findall(re.compile('\[[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('\[[a-z]\]'), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])

    if (len(re.findall(re.compile("[0-9]{2}'[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]{2}'[a-z]"), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])
                tokens.insert(i+2,tok[3])

    if (len(re.findall(re.compile("[0-9]'[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]'[a-z]"), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])

    if (len(re.findall(re.compile("[0-9]{2}[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]{2}[a-z]"), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])

    if (len(re.findall(re.compile("[0-9][a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9][a-z]"), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])

    if (len(re.findall(re.compile("\.\d+,\d+"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("\.\d+,\d+"), tok):
                result = add_symbol(tok)
                tokens[i] = result[0]
                for k, res in enumerate(result[1:], start=1):
                    tokens.insert(i+k,res)

    if check_correct_tokenize(iupac, tokens) == True:
        return tokens
    else:
        return None

################################################################

from tqdm import tqdm
import torch.multiprocessing as mp
import time
def map_parallel (lst, fn, nworkers=1, fallback_sharing=False):
    if nworkers == 1:
        return [fn(item) for item in lst]

    if fallback_sharing:
        mp.set_sharing_strategy('file_system')
    L = mp.Lock()
    QO = mp.Queue(nworkers)
    QI = mp.Queue()
    for item in lst:
        QI.put(item)

    def pfn ():
        time.sleep(0.0001)
        #print(QI.empty(), QI.qsize())
        while QI.qsize() > 0: #not QI.empty():
            L.acquire()
            item = QI.get()
            L.release()
            obj = fn(item)
            QO.put(obj)

    procs = []
    for nw in range(nworkers):
        P = mp.Process(target=pfn, daemon=True)
        time.sleep(0.0001)
        P.start()
        procs.append(P)

    return [QO.get() for i in range(len(lst))]


def reverse_vocab (vocab):
    return dict((v,k) for k,v in vocab.items())

class IupacModel ():
    def __init__ (self, names, nworkers=1):
        self.special_tokens =  ['<pad>', '<unk>', '<bos>', '<eos>', ';', '.', '>']
        vocab = set()
        names = map_parallel(names, iupac_tokenizer, nworkers)
        for i in tqdm(range(len(names))):
            tokens = names[i]
            if tokens is not None:
                tokens = set(tokens)
                tokens -= set(self.special_tokens)
                vocab |= tokens
        nspec = len(self.special_tokens)
        self.vocab = dict(zip(sorted(vocab),
                              range(nspec, nspec+len(vocab))))
        for i,spec in enumerate(self.special_tokens):
            self.vocab[spec] = i
        self.rev_vocab = reverse_vocab(self.vocab)
        self.vocsize = len(self.vocab)

    def encode (self, seq):
        tokens = iupac_tokenizer(seq)
        if tokens is None:
            raise Exception(f"Unable to tokenize IUPAC name: {seq}")
        else:
            return [self.vocab[t] for t in tokens]

    def decode (self, seq):
        seq = [s for s in seq if s >= 4]
        return "".join([self.rev_vocab[code] for code in seq])
