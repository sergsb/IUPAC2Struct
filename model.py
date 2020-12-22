import torch
import numpy as np
from tqdm import tqdm

def subsequent_mask (tgt_mask):
    size = tgt_mask.size(-1)
    return tgt_mask.to(torch.uint8) & torch.tril(torch.ones(1,size,size, dtype=torch.uint8)).to(tgt_mask.device)

class DataParallel (torch.nn.Module):
    def __init__(self, model):
        super(DataParallel, self).__init__()
        self.model = torch.nn.DataParallel(model).cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

class Model:
    def __init__ (self):
        pass

    def predict_single (self, inp, beam=1, temp=0, naug=0):
        inp = self.src_model.encode(inp)
        naug = 0
        naug += 1
        with torch.no_grad():
            self.T.eval()

            src = []
            for b in range(beam):
                src += [inp]

            src = pad_pack(src).to(self.device)
            src_mask = (src != 0).unsqueeze(-2).to(self.device)
            src_mem = self.T.encoder(self.T.src_embedder(src), src_mask)

            cands = [[2] for b in range(beam)]
            probs = [[] for b in range(beam)]
            res_probs = []
            res = []

            for step in range(500):
                if beam <= 0:
                    break

                aug_cands = []
                for c in cands:
                    aug_cands += [c]*naug

                tgt = pad_pack(aug_cands).to(self.device)
                tgt_mask = subsequent_mask((tgt != 0).unsqueeze(-2)).type_as(src_mask)
                out = self.T.decoder(self.T.tgt_embedder(tgt), src_mem, src_mask, tgt_mask)
                out = self.T.generator(out)[:,-1]

                out = out.view(beam, naug, out.shape[-1])
                out = out.mean(dim=1)

                pbs,wds = torch.sort(out, dim=1, descending=True)
                step_cands = []
                step_probs = []
                for i in range(beam):
                    for j in range(beam*2):
                        step_cands.append(cands[i]+[wds[i,j].tolist()])
                        step_probs.append(probs[i]+[pbs[i,j].tolist()])
                step_cands, step_probs = remove_duplicates(step_cands, step_probs)
                best_ids = np.argsort([prob_score(pb) for pb in step_probs])[::-1][:beam]

                cands = []
                probs = []
                for i in best_ids:
                    if step_cands[i][-1] == 3:
                        res.append(step_cands[i])
                        res_probs.append(step_probs[i])
                        beam -= 1
                        src_mem = src_mem[:-1*naug]
                        src_mask = src_mask[:-1*naug]
                        src = src[:-1*naug]
                    else:
                        cands.append(step_cands[i])
                        probs.append(step_probs[i])

        if beam > 0:
            res += cands[:beam]
            res_probs += probs[:beam]


        pred = []
        for r in res:
            try:
                pred.append(self.tgt_model.decode([s for s in r if s >= 4]))
            except:
                pred.append(None)

        pred_probs = [prob_score(pb) for pb in res_probs]
        order = np.argsort(pred_probs)[::-1]
        final_outputs = np.array(pred)[order].tolist()
        final_probs = np.array(pred_probs)[order].tolist()
        return final_outputs, final_probs

    def predict (self, X, beam=1, temp=0, naug=0):
        pred = [[None for i in range(len(X))] for b in range(beam)]
        for i in tqdm(range(len(X))):
            cands, probs = self.predict_single(X[i], beam=beam, temp=temp, naug=naug)
            if len(cands) > 0:
                for b in range(beam):
                    pred[b][i] = cands[b]
        return pred

def prob_score (prob):
    return np.exp(np.mean(prob))

def remove_duplicates (cands, probs):
    use_cands = []
    use_probs = []
    for i in range(len(cands)):
        if cands[i] not in use_cands:
            use_cands.append(cands[i])
            use_probs.append(probs[i])
    return use_cands, use_probs

def pad_pack (sequences):
    maxlen = max(map(len, sequences))
    batch = torch.LongTensor(len(sequences),maxlen).fill_(0)
    for i,x in enumerate(sequences):
        batch[i,:len(x)] = torch.LongTensor(x)
    return batch
