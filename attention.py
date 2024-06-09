import torch
import torch.nn as nn

from check_dimensions import CheckDimensions
from params import *


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        self.query = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        # TODO: shouldn't value be from EMBEDDING_SIZE to EMBEDDING_SIZE so that the output has dim BxTxC (giving the delta-E for changing the embedding vector)
        # Likely as this is only the value-down part.
        self.valueDown = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE])
    def keys(self, input):
        return self.key(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE])
    def queries(self, input):
        return self.query(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE])
    def dotProductKeysAndQueriesAndScale(self, keys, queries):
        return queries @ keys.transpose(-2,-1) * HEAD_SIZE**-0.5

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE])
    def applyDropout(self, input):
        return self.dropout(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE])
    def softmaxMasked(self, weights):
        weightsMasked = weights.masked_fill(self.tril[:BLOCK_SIZE, :BLOCK_SIZE] == 0, float('-inf'))
        return nn.functional.softmax(weightsMasked, dim = -1)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE])
    def dropoutSoftmaxMasked(self, weights):
        return self.applyDropout(self.softmaxMasked(weights))

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE])
    def values(self, input):
        return self.valueDown(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE])
    def scaleValuesByWeights(self, values, weights):
        return weights @ values

    def forward(self, input_BTC):
        keys = self.keys(input_BTC)
        queries = self.queries(input_BTC)
        values = self.values(input_BTC)

        weights = self.dotProductKeysAndQueriesAndScale(keys, queries)
        weightsDropoutSoftmaxMasked = self.dropoutSoftmaxMasked(weights)

        return self.scaleValuesByWeights(values, weightsDropoutSoftmaxMasked)


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(NUM_HEADS)])
        # TODO: what does this projection achieve (see 1:31 of Andrej's vid)
        # Probably value-up transformation
        self.projection = nn.Linear(HEAD_SIZE * NUM_HEADS, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE * NUM_HEADS])
    def headsOutput(self, input):
        return torch.cat([head(input) for head in self.heads], dim = -1)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def projectToEmbeddingSpace(self, input):
        return self.projection(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def applyDropout(self, input):
        return self.dropout(input)

    def forward(self, input_BTC):
        headsOutput = self.headsOutput(input_BTC)
        headsOutputProjected = self.projectToEmbeddingSpace(headsOutput)
        return self.applyDropout(headsOutputProjected)

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Apparently a step-up by factor of 4 in dimension is optimal (from attention paper)
            nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE),
            nn.ReLU(),
            # Not fully sure why this extra linear layer is needed, 1:30 in Karpathy's vid for context
            nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.Dropout(DROPOUT)
        )

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def layer(self, input):
        return self.net(input)

    def forward(self, input_BTC):
        return self.net(input_BTC)

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        super().__init__()
        # Andrej does some shenanigans here using a head_size = embed_size // num_heads - are we ok to ignore?
        self.multiHeadAttention = MultiHeadAttention()
        self.feedForward = FeedForward()
        self.layerNorm1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.layerNorm2 = nn.LayerNorm(EMBEDDING_SIZE)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def multiHeadAttentionWrapper(self, input):
        return self.multiHeadAttention(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def feedForwardWrapper(self, input):
        return self.feedForward(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def layerNorm1Wrapper(self, input):
        return self.layerNorm1(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def layerNorm2Wrapper(self, input):
        return self.layerNorm2(input)

    def forward(self, input_BTC):
        # Cannot use += as this modifies in-place and interrupts gradient computation
        input_BTC = input_BTC + self.multiHeadAttentionWrapper(self.layerNorm1Wrapper(input_BTC))
        input_BTC = input_BTC + self.feedForwardWrapper(self.layerNorm2Wrapper(input_BTC))
        return input_BTC


class Transformer(nn.Module):
    """ Multiple transformer blocks in series """

    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.positionEmbeddingTable = nn.Embedding(BLOCK_SIZE, EMBEDDING_SIZE)
        self.transformerBlocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.finalLayerNorm = nn.LayerNorm(EMBEDDING_SIZE)
        self.logits = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def getTokenEmbedding(self, input):
        return self.tokenEmbeddingTable(input)

    @CheckDimensions([BLOCK_SIZE, EMBEDDING_SIZE])
    def getPositionEmbedding(self):
        return self.positionEmbeddingTable(torch.arange(BLOCK_SIZE, device="mps"))

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def transformerBlocksWrapper(self, input):
        return self.transformerBlocks(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, EMBEDDING_SIZE])
    def finalLayerNormWrapper(self, input):
        return self.finalLayerNorm(input)

    @CheckDimensions([BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE])
    def getLogits(self, input):
        return self.logits(input)

    def forward(self, inputs_BT, targets_BT=None):
        tokenEmbeddings = self.getTokenEmbedding(inputs_BT)
        positionEmbeddings = self.getPositionEmbedding()

        inputs = tokenEmbeddings + positionEmbeddings
        inputs = self.transformerBlocksWrapper(inputs)
        inputs = self.finalLayerNormWrapper(inputs)

        logits = self.getLogits(inputs)

        if targets_BT is None:
            loss = None
        else:
            logitsReworked = logits.view(BATCH_SIZE * BLOCK_SIZE, VOCAB_SIZE)
            targetsReworked = targets_BT.view(BATCH_SIZE * BLOCK_SIZE)
            loss = nn.functional.cross_entropy(logitsReworked, targetsReworked)

        return logits, loss

    @CheckDimensions([BATCH_SIZE, VOCAB_SIZE])
    def convertLogitsToProbabilities(self, logits):
        return nn.functional.softmax(logits, dim = -1)

    @CheckDimensions([BATCH_SIZE, 1])
    def sampleNewTokens(self, probabilities):
        return torch.multinomial(probabilities, num_samples = 1)

    def generate(self, context_BT, numTokensToGenerate):
        newTokens = torch.tensor([], device="mps")
        for _ in range(numTokensToGenerate):
            logits, _ = self(context_BT)
            logitsOfFinalTokens = logits[:, -1, :]
            probabilities = self.convertLogitsToProbabilities(logitsOfFinalTokens)
            newTokens = torch.cat((newTokens, self.sampleNewTokens(probabilities)), dim = 1)
        return torch.cat((context_BT, newTokens), dim = 1)
