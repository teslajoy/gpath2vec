import networkx as nx
import torch
import random
import numpy as np


class Embedder:
    def __init__(self):
        self.model = self.method()

    @staticmethod
    def method():
        return None


class PathwayMetapath2vec(Embedder):
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name
        self.embeddings = {}
        self.vocab = {}
        super().__init__()

    def method(self):
        return self.metapath2vec()

    def metapath2vec(self):
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        node_types = nx.get_node_attributes(self.graph, "node_type")
        pathway_metapaths = [["sig", "sig"], ["sig", "notsig", "sig"], ["notsig", "sig", "notsig"]]
        walks = []
        random.seed(1234)

        for start_node in self.graph.nodes():
            walk = [start_node]
            current = start_node
            for i in range(100):
                neighbors = list(self.graph.neighbors(current))
                if not neighbors: break
                metapath_idx = i % len(random.choice(pathway_metapaths))
                target_type = random.choice(pathway_metapaths)[metapath_idx]
                typed_neighbors = [n for n in neighbors if node_types.get(n, "unknown") == target_type]
                next_node = random.choice(typed_neighbors) if typed_neighbors else random.choice(neighbors)
                walk.append(next_node)
                current = next_node
            walks.append(walk)

        print(f"Number of random walks: {len(walks)}")
        return walks

    def train_embeddings(self, walks, dimensions=128, window_size=5, epochs=10, lr=0.025):
        word_to_ix = {}
        for walk in walks:
            for word in walk:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        vocab_size = len(word_to_ix)
        self.vocab = word_to_ix

        class SkipGram(torch.nn.Module):
            def __init__(self, vocab_size, embedding_dim):
                super().__init__()
                self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
                self.output = torch.nn.Embedding(vocab_size, embedding_dim)
                self.embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
                self.output.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

            def forward(self, target, context):
                return torch.sum(self.embeddings(target) * self.output(context), dim=1)

        model = SkipGram(vocab_size, dimensions)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        training_data = []
        for walk in walks:
            for i, word in enumerate(walk):
                target_ix = word_to_ix[word]
                context_indices = list(range(max(0, i - window_size), i)) + list(
                    range(i + 1, min(len(walk), i + window_size + 1)))
                training_data.extend([(target_ix, word_to_ix[walk[ctx_i]]) for ctx_i in context_indices])

        batch_size = 512
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                target_batch = torch.LongTensor([pair[0] for pair in batch])
                context_batch = torch.LongTensor([pair[1] for pair in batch])
                neg_contexts = torch.LongTensor(np.random.choice(vocab_size, size=(len(batch), 5)).tolist())

                model.zero_grad()
                pos_loss = torch.nn.functional.logsigmoid(model(target_batch, context_batch))
                neg_loss = sum(
                    torch.nn.functional.logsigmoid(-model(target_batch, neg_contexts[:, j])) for j in range(5))
                loss = -torch.mean(pos_loss + neg_loss)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        self.model = model
        with torch.no_grad():
            self.embeddings = {word: model.embeddings.weight[idx].numpy() for word, idx in word_to_ix.items()}

        return model

    def get_embeddings(self):
        return self.embeddings

    def save_model(self, path):
        torch.save({'embeddings': self.embeddings, 'vocab': self.vocab}, path)

    def load_model(self, path):
        state = torch.load(path)
        self.embeddings = state['embeddings']
        self.vocab = state['vocab']