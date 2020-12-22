#from utils import *
import re
import torch
from tqdm import tqdm

def smiles_atom_tokenizer (smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def reverse_vocab (vocab):
    return dict((v,k) for k,v in vocab.items())

class SmilesModel ():
    def __init__ (self, smiles, naug=3):
        if type(smiles) is str:
            smiles = extract_smiles(smiles)
        
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '>', '.']
        vocab = set()
        for i in tqdm(range(len(smiles))):
            sm = smiles[i]
            for n in range(naug):
                tokens = smiles_atom_tokenizer(augment_smile(sm))
                vocab |= set(tokens)
        nspec = len(self.special_tokens)
        self.vocab = dict(zip(sorted(vocab),
                              range(nspec, nspec+len(vocab))))
        for i,spec in enumerate(self.special_tokens):
            self.vocab[spec] = i
        self.rev_vocab = reverse_vocab(self.vocab)
        self.vocsize = len(self.vocab)
    
    def encode (self, seq):
        return [self.vocab[token] for token in smiles_atom_tokenizer(seq)]
        
    def decode (self, seq):
        seq = [s for s in seq if s >= 4]
        return "".join([self.rev_vocab[code] for code in seq])

class RawSmilesModel ():
    def __init__ (self, smiles):
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '>', '.']
        vocab = set()
        for i in tqdm(range(len(smiles))):
            sm = smiles[i]
            chars = set(list(sm))
            chars -= set(self.special_tokens)
            vocab |= chars
        nspec = len(self.special_tokens)
        self.vocab = dict(zip(sorted(vocab),
                              range(nspec, nspec+len(vocab))))
        for i,spec in enumerate(self.special_tokens):
            self.vocab[spec] = i
        self.rev_vocab = reverse_vocab(self.vocab)
        self.vocsize = len(self.vocab)
    
    def encode (self, seq):
        return [self.vocab[char] for char in seq]
        
    def decode (self, seq):
        seq = [s for s in seq if s >= 4]
        return "".join([self.rev_vocab[code] for code in seq])


if __name__ == "__main__":
    smiles = extract_smiles("../data/maximal.csv")

    print('Building Raw Smiles model')
    sys.stdout.flush()
    RSM = RawSmilesModel(smiles)
    torch.save(RSM, "raw_smiles_model.pt")

    print('Building Smiles model')
    sys.stdout.flush()
    SM = SmilesModel(smiles)
    torch.save(SM, "smiles_model.pt")
