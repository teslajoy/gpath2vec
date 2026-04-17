"""metapath2vec embeddings for pathway graphs."""

import random

import networkx as nx
import numpy as np
import torch


class Embedder:
    def __init__(self):
        self.model = self.method()

    @staticmethod
    def method():
        return None


class PathwayMetapath2vec(Embedder):
    def __init__(self, graph, name, walks_per_node=10, walk_length=100):
        self.graph = graph
        self.name = name
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.embeddings = {}
        self.vocab = {}
        super().__init__()

    def method(self):
        return self.metapath2vec()

    def _weighted_choice(self, neighbors, current):
        """sample a neighbor proportional to edge weight, uniform if no weights."""
        weights = []
        for n in neighbors:
            e = self.graph.edges[current, n] if self.graph.has_edge(current, n) else {}
            weights.append(e.get("weight", 1.0))
        total = sum(weights)
        if total == 0:
            return random.choice(neighbors)
        probs = [w / total for w in weights]
        return random.choices(neighbors, weights=probs, k=1)[0]

    def metapath2vec(self):
        print(f"graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        node_types = nx.get_node_attributes(self.graph, "node_type")

        metapaths = [
            ["sig", "sig"],
            ["sig", "notsig", "sig"],
            ["notsig", "sig", "notsig"],
            ["cluster", "sig", "sig"],
            ["cluster", "sig", "notsig"],
            ["cluster", "notsig", "sig"],
        ]

        walks = []
        random.seed(1234)

        for _ in range(self.walks_per_node):
            for start_node in self.graph.nodes():
                walk = [start_node]
                current = start_node
                mp = random.choice(metapaths)

                for i in range(self.walk_length):
                    neighbors = list(self.graph.neighbors(current))
                    if not neighbors:
                        break

                    target_type = mp[i % len(mp)]
                    typed = [n for n in neighbors if node_types.get(n, "unknown") == target_type]

                    if typed:
                        next_node = self._weighted_choice(typed, current)
                    else:
                        next_node = self._weighted_choice(neighbors, current)

                    walk.append(next_node)
                    current = next_node

                walks.append(walk)

        print(f"random walks: {len(walks)}")
        return walks

    def train_embeddings(self, walks, dimensions=512, window_size=5, epochs=10, lr=0.025):
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # build training pairs
        training_data = []
        for walk in walks:
            for i, word in enumerate(walk):
                target_ix = word_to_ix[word]
                ctx_start = max(0, i - window_size)
                ctx_end = min(len(walk), i + window_size + 1)
                for j in range(ctx_start, ctx_end):
                    if j != i:
                        training_data.append((target_ix, word_to_ix[walk[j]]))

        batch_size = 1024
        n_neg = 5
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                targets = torch.LongTensor([p[0] for p in batch])
                contexts = torch.LongTensor([p[1] for p in batch])
                neg = torch.LongTensor(
                    np.random.choice(vocab_size, size=(len(batch), n_neg)).tolist()
                )

                optimizer.zero_grad()
                pos_score = torch.nn.functional.logsigmoid(model(targets, contexts))
                neg_score = sum(
                    torch.nn.functional.logsigmoid(-model(targets, neg[:, j]))
                    for j in range(n_neg)
                )
                loss = -torch.mean(pos_score + neg_score)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            n_batches = max(1, len(training_data) // batch_size)
            print(f"epoch {epoch + 1}/{epochs}, loss: {total_loss / n_batches:.4f}")

        self.model = model
        with torch.no_grad():
            self.embeddings = {
                word: model.embeddings.weight[idx].numpy()
                for word, idx in word_to_ix.items()
            }
        return model

    def get_embeddings(self):
        return self.embeddings

    def save_model(self, path):
        torch.save({"embeddings": self.embeddings, "vocab": self.vocab}, path)

    def load_model(self, path):
        state = torch.load(path)
        self.embeddings = state["embeddings"]
        self.vocab = state["vocab"]
