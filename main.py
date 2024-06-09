from attention import Transformer
import torch

from params import *
from timing import timing

TRAIN = "train"
VAL = "val"

@timing
def trainModel():

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in datasets.keys():
            losses = torch.zeros(LOSS_EVAL_ITERS)
            for k in range(LOSS_EVAL_ITERS):
                X, Y = getBatch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = Transformer().to("mps")
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    optimiser = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    for step in range(NUM_STEPS):
        inputsBatch, targetsBatch = getBatch(TRAIN)
        logits, loss = model(inputsBatch, targetsBatch)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if step % LOSS_EVAL_INTERVAL == 0 or step == NUM_STEPS - 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    return model

def getBatch(split):
    data = datasets[split]
    startingIndexes_B = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    inputs_BT = torch.stack([data[startingIndex : startingIndex + BLOCK_SIZE] for startingIndex in startingIndexes_B])
    targets_BT = torch.stack([data[startingIndex + 1 : startingIndex + BLOCK_SIZE + 1] for startingIndex in startingIndexes_B])
    return inputs_BT.to("mps"), targets_BT.to("mps")


# TODO: look into memory usage and see if wrapping things in functions would make matrix operations more memory efficient
# TODO: make everything use the GPU

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(len(data))
n = int(TRAIN_VAL_SPLIT * len(data))
datasets = {
    TRAIN : data[:n],
    VAL : data[n:]
}

# trainData = torch.tensor([0,2,4,5,7,9,11,9,7,4,5,4,0,2,7,4,0,4,7,9,4,9,7,2,4,7,11,7,4,0])
# print(trainData)

model = trainModel()
print(model.generate(torch.zeros((12, 8), dtype=torch.long, device="mps"), 20))
