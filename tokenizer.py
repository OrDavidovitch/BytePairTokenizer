from collections import Counter
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import string
import os

def mostCommonPairOneCore(textArray, allowedCodes):
    first = textArray[:-1]
    second = textArray[1:]
    pairs = np.vstack((first, second)).T

    # Filter out pairs where either element corresponds to a space
    valid_mask = np.isin(pairs[:, 0], allowedCodes) & np.isin(pairs[:, 1], allowedCodes)
    pairs = pairs[valid_mask]

    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    mostCommon = unique_pairs[counts.argmax()]
    return mostCommon

def updateTextOneCore(textArray, mostCommonPair, newTokenId):
    first = textArray[:-1]
    second = textArray[1:]
    mask = (first == mostCommonPair[0]) & (second == mostCommonPair[1])

    # Precompute output length
    outputLength = len(first) + 1 - np.count_nonzero(mask)

    # Prepare output array
    output = np.empty(outputLength, dtype=np.int32)

    # Compute positions in output
    # Position offsets: count cumulative non-merged positions
    mergeCumSum = np.cumsum(mask)

    # Assign non-merged tokens
    nonMergePositions = np.flatnonzero(~mask)
    nonMergePositionsOutput = nonMergePositions - mergeCumSum[nonMergePositions]
    output[nonMergePositionsOutput] = first[nonMergePositions]

    # Assign merged tokens
    mergePositions = np.flatnonzero(mask) ### all indices where a mostCommonPair starts
    mergePositionsOutput = mergePositions - mergeCumSum[mergePositions] + 1
    output[mergePositionsOutput] = newTokenId


    if mask[-1] == 0:
        output[-1] = second[-1]
    return output
    
    

def countPairs(shmName, shape, dtype, start, end, allowedCodes):
    ### Excess the shared memory
    existingShm = shared_memory.SharedMemory(name=shmName)    
    ### Build a Numpy array of the chunk
    data = np.ndarray(shape, dtype=dtype, buffer=existingShm.buf)
    ### The end paramter can be larger than actual end 
    actual_end = min(end, len(data) - 1) 
    
    ### Building an array of all subsequent pairs
    first = data[start:actual_end - 1]
    second = data[start + 1:actual_end]
    pairs = np.vstack((first, second)).T

    ### Filter out pairs where either element corresponds 
    ### to non-letters or merged letters
    valid_mask = np.isin(pairs[:, 0], allowedCodes) & np.isin(pairs[:, 1], allowedCodes)
    pairs = pairs[valid_mask]
    
    ### Count the pairs and build a corresponding Counter object
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    result = Counter({(int(a), int(b)): int(c) for (a, b), c in zip(unique_pairs, counts)})
    existingShm.close()
    return result

def updateTextChunk(shmName, shape, dtype, start, end, mostCommonPair, newTokenId):
    """
    Replaces each appearnce of the most common pair with a new token
    Returns : Updated shared array and updated text as a Numpy array
    """
        
    ### Excess shared memory
    existingShm = shared_memory.SharedMemory(name=shmName)
    data = np.ndarray(shape, dtype=dtype, buffer=existingShm.buf)
        
    ### Create a mask pointing the appearences of most common pair
    first = data[start:end-1]
    second = data[start+1:end]
    mask = (first == mostCommonPair[0]) & (second == mostCommonPair[1])

    ### Precompute output length
    outputLength = len(first) + 1 - np.count_nonzero(mask)

    ### Prepare output array
    output = np.empty(outputLength, dtype=np.int32)

    ### Compute positions in output
    ### Position offsets: count cumulative non-merged positions
    mergeCumSum = np.cumsum(mask)

    ### Assign non-merged tokens
    nonMergePositions = np.flatnonzero(~mask)
    nonMergePositionsOutput = nonMergePositions - mergeCumSum[nonMergePositions]
    output[nonMergePositionsOutput] = first[nonMergePositions]

    ### Assign merged tokens
    mergePositions = np.flatnonzero(mask) ### all indices where a mostCommonPair starts
    mergePositionsOutput = mergePositions - mergeCumSum[mergePositions] + 1
    output[mergePositionsOutput] = newTokenId

    if mask[-1] == 0:
        output[-1] = second[-1]

    existingShm.close()
    return output


class bytePairTokenizer:
    def __init__(self, unkToken = "UNK", numberOfStartingTokens = 256):
        self.vocab = {i: chr(i) for i in range(numberOfStartingTokens)} ### Token ID : Token str
        self.inverseVocab = {chr(i) : i for i in range(numberOfStartingTokens)} ### Token str : Token ID
        self.nextTokenId = numberOfStartingTokens
        self.unkToken = unkToken
        self.vocab[self.nextTokenId] = self.unkToken
        self.inverseVocab[unkToken] = self.nextTokenId
        self.nextTokenId += 1
        

    def normalize(self, text):
        txt = text.lower()
        txt = txt.replace('’', "'")
        txt = txt.replace('“', '"')
        txt = txt.replace('”', '"')
        txt = txt.replace('‘', "'")
        return txt 

    def textToSharedArray(self, text):
        ### convert text to integer array using utf-8
        txt = self.normalize(text)
        arr = np.array([self.inverseVocab.get(char, self.inverseVocab[self.unkToken]) for char in txt], dtype=np.int32)
        ### create shared memory
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        ### copy array to shared memory
        sharedArray = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        sharedArray[:] = arr[:]
        return shm, sharedArray

    
    def mostCommonPair(self, shm, sharedArray, allowedCodes, numWorkers=4):
            
        length = len(sharedArray)
        chunkSize = length // numWorkers
        args = [(shm.name, sharedArray.shape, sharedArray.dtype, 
                 i * chunkSize, (i + 1) * chunkSize + 1, allowedCodes) for i in range(numWorkers)]
        with mp.Pool(processes=numWorkers) as pool:
            results = pool.starmap(countPairs, args)
        totalCounts = Counter()
        for counter in results:
            totalCounts.update(counter)
        if not totalCounts:
            return None  # No pairs left to merge
        return totalCounts.most_common(1)[0][0]



    def updateText(self, shm, sharedArray, mostCommonPair, newTokenId, numWorkers=4):
        length = len(sharedArray)
        chunkSize = length // numWorkers
        args = [(shm.name, sharedArray.shape, sharedArray.dtype,
                 i * chunkSize, min((i + 1) * chunkSize + 1, length), mostCommonPair, newTokenId) for i in range(numWorkers)]
        with mp.Pool(processes=numWorkers) as pool:
            results = pool.starmap(updateTextChunk, args)

        newText = np.concatenate(results)
        
        newSharedArray = np.ndarray(newText.shape, dtype=newText.dtype, buffer=shm.buf)
        newSharedArray[:] = newText[:]
        return shm, newSharedArray

    def train(self, text, numMerges=1000, numWorkers=4):
        
        chars = string.ascii_letters[:26]
        allowedCodes = np.array([ord(ch) for ch in chars])
        merges = 0 
        if numWorkers == 1:
            txt = self.normalize(text)
            textArray = np.array([ord(char) for char in txt])
            charsAndPunc = string.ascii_letters[:26]
            allowedCodes = np.array([ord(ch) for ch in chars])
            while merges < numMerges:
                print(f"Merges Created: {merges}")
                mostCommonPair = mostCommonPairOneCore(textArray, allowedCodes)
                print(f"Most Common Pair Ids: {mostCommonPair}")
                print(f"Most Common Pair: {self.vocab[mostCommonPair[0]] + self.vocab[mostCommonPair[1]]}")
                newTokenStr = self.vocab[mostCommonPair[0]] + self.vocab[mostCommonPair[1]]
                newTokenId = self.nextTokenId
                self.vocab[newTokenId] = newTokenStr
                self.inverseVocab[newTokenStr] = newTokenId
                self.nextTokenId += 1
                textArray = updateTextOneCore(textArray, mostCommonPair, newTokenId)
                allowedCodes = np.append(allowedCodes, newTokenId)
                merges += 1

        else:
            
            shm, sharedArray = self.textToSharedArray(text)
            try:
                while merges < numMerges:
                    print(f"Merges Created: {merges}")
                    mostCommonPair = self.mostCommonPair(shm, sharedArray, allowedCodes, numWorkers)
                    print(f"Most Common Pair Ids: {mostCommonPair}")
                    print(f"Most Common Pair: {self.vocab[mostCommonPair[0]] + self.vocab[mostCommonPair[1]]}")
                    if not mostCommonPair:
                        print("No more pairs to merge.")
                        break
                    newTokenStr = self.vocab[mostCommonPair[0]] + self.vocab[mostCommonPair[1]]
                    newTokenId = self.nextTokenId
                    self.vocab[newTokenId] = newTokenStr
                    self.inverseVocab[newTokenStr] = newTokenId
                    self.nextTokenId += 1
                    shm, sharedArray = self.updateText(shm, sharedArray, mostCommonPair, newTokenId, numWorkers)
                    allowedCodes = np.append(allowedCodes, newTokenId)
                    merges += 1
            finally:
                shm.close()
                shm.unlink()

    def tokenize(self, text):
        tokens = []
        i = 0
        max_token_length = max(len(token) for token in self.inverseVocab)
        while i < len(text):
            matched = False
            for length in range(min(len(text) - i, max_token_length), 0, -1):
                substr = text[i:i+length]
                if substr in self.inverseVocab:
                    tokens.append(self.inverseVocab[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1
        return tokens


    def decode(self, tokenIds):
        return ''.join(self.vocab[tokenId] for tokenId in tokenIds)
        
