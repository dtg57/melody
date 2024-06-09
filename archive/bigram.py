import torch
import torch.nn as nn


LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 6
BATCH_SIZE = 8  # B
BLOCK_SIZE = 8  # lookbehind distance (irrelevant for bigram) T
VOCAB_SIZE = 12
NUM_STEPS = 1000

trainData = torch.tensor([0,2,4,5,7,9,11,9,7,4,5,4,0,2,7,4,0,4,7,9,4,9,7,2,4,7,11,7,4,0])

class BigramModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embeddingTable = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)

    def forward(self, inputs_BT, targets_BT=None):

        logits_BTC = self.embeddingTable(inputs_BT)
        if targets_BT is None:
            loss = None
        else:
            B, T, C = logits_BTC.shape
            logitsReworked = logits_BTC.view(B * T, C)
            targetsReworked = targets_BT.view(B * T)
            print(logitsReworked.shape)
            print(targetsReworked.shape)
            loss = nn.functional.cross_entropy(logitsReworked, targetsReworked)

        return logits_BTC, loss

    def generate(self, context_BT, numTokensToGenerate):
        newTokens_Bn = torch.tensor([])
        for _ in range(numTokensToGenerate):
            logits_BTC, _ = self(context_BT)
            logits_BC = logits_BTC[:, -1, :]
            probabilities_BC = nn.functional.softmax(logits_BC, dim = -1)
            newTokens_B = torch.multinomial(probabilities_BC, num_samples = 1)
            newTokens_Bn = torch.cat((newTokens_Bn, newTokens_B), dim = 1)
        return torch.cat((context_BT, newTokens_Bn), dim = 1)

# trainData = torch.randint(VOCAB_SIZE, (1000,))
print(trainData)

def getBatch():
    startingIndexes_B = torch.randint(len(trainData) - BLOCK_SIZE, (BATCH_SIZE,))
    inputs_BT = torch.stack([trainData[startingIndex : startingIndex + BLOCK_SIZE] for startingIndex in startingIndexes_B])
    targets_BT = torch.stack([trainData[startingIndex + 1 : startingIndex + BLOCK_SIZE + 1] for startingIndex in startingIndexes_B])
    return inputs_BT, targets_BT

@timing
def trainModel():
    model = BigramModel().to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    for step in range(NUM_STEPS):
        inputsBatch, targetsBatch = getBatch()
        logits, loss = model(inputsBatch.to("mps"), targetsBatch.to("mps"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step % 100 == 0):
            print("Loss at step ", step, loss.item())
    return model

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

model = trainModel()
for i in range(10):
    print(model.generate(torch.tensor([[0,2,4,5,7]]).to("mps"), 10))
