"""embedding methods for pathway graphs."""

import random

import networkx as nx
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding as SklearnSpectral


class Embedder:
    """base class for all embedding methods."""

    def __init__(self):
        self.embeddings = {}
        self.model = self.method()

    @staticmethod
    def method():
        return None

    def get_embeddings(self):
        return self.embeddings

    def save_model(self, path):
        torch.save({"embeddings": self.embeddings}, path)

    def load_model(self, path):
        state = torch.load(path)
        self.embeddings = state["embeddings"]


class VAEEmbedder(Embedder):
    """
    variational autoencoder on the ea matrix (cluster x pathway).
    learns a smooth latent space of pathway activity patterns.

    ea_matrix: pandas dataframe (cluster x pathway)
    dimensions: latent space size
    hidden_dim: hidden layer size
    epochs: training epochs
    lr: learning rate
    beta: weight on kl divergence (beta-vae)
    """

    def __init__(self, ea_matrix, dimensions=512, hidden_dim=256,
                 epochs=50, lr=0.001, beta=1.0):
        self.ea_matrix = ea_matrix
        self.dimensions = dimensions
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.latent_means = {}
        self.latent_vars = {}
        super().__init__()

    def method(self):
        input_dim = self.ea_matrix.shape[1]
        X = torch.FloatTensor(self.ea_matrix.values)
        names = list(self.ea_matrix.index)

        class VAE(torch.nn.Module):
            def __init__(vae, input_dim, hidden_dim, latent_dim):
                super().__init__()
                vae.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                )
                vae.mu = torch.nn.Linear(hidden_dim, latent_dim)
                vae.logvar = torch.nn.Linear(hidden_dim, latent_dim)
                vae.decoder = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, input_dim),
                    torch.nn.Sigmoid(),
                )

            def encode(vae, x):
                h = vae.encoder(x)
                return vae.mu(h), vae.logvar(h)

            def reparameterize(vae, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(vae, z):
                return vae.decoder(z)

            def forward(vae, x):
                mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)
                return vae.decode(z), mu, logvar

        model = VAE(input_dim, self.hidden_dim, self.dimensions)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        print(f"vae: {len(names)} clusters, {input_dim} pathways -> {self.dimensions} latent dims")

        batch_size = min(256, len(names))
        for epoch in range(self.epochs):
            perm = torch.randperm(len(names))
            total_loss = 0
            n_batches = 0

            for i in range(0, len(names), batch_size):
                batch = X[perm[i:i + batch_size]]
                recon, mu, logvar = model(batch)

                recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg = total_loss / len(names)
                print(f"  epoch {epoch+1}/{self.epochs}, loss: {avg:.4f}")

        # extract latent means as embeddings
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(X)
            embeddings_np = mu.numpy()
            vars_np = logvar.exp().numpy()

        self.embeddings = {names[i]: embeddings_np[i] for i in range(len(names))}
        self.latent_means = self.embeddings
        self.latent_vars = {names[i]: vars_np[i] for i in range(len(names))}
        self._model = model
        return model

    def get_uncertainty(self):
        """return per-cluster latent variance (uncertainty estimate)."""
        return self.latent_vars

    def save_model(self, path):
        """
        saves: embeddings (latent means), latent_vars (uncertainty),
        and model weights (for generation via model.decode(z)).
        """
        torch.save({
            "embeddings": self.embeddings,
            "latent_vars": self.latent_vars,
            "model_state": self._model.state_dict() if hasattr(self, "_model") else None,
        }, path)

    def load_model(self, path):
        state = torch.load(path)
        self.embeddings = state["embeddings"]
        self.latent_vars = state.get("latent_vars", {})


class SVDEmbedder(Embedder):
    """
    truncated svd on the ea matrix (cluster x pathway).
    baseline: no graph structure, just the enrichment signal.

    ea_matrix: pandas dataframe (cluster x pathway)
    dimensions: output embedding size
    """

    def __init__(self, ea_matrix, dimensions=512):
        self.ea_matrix = ea_matrix
        self.dimensions = min(dimensions, min(ea_matrix.shape) - 1)
        super().__init__()

    def method(self):
        svd = TruncatedSVD(n_components=self.dimensions, random_state=42)
        X = svd.fit_transform(self.ea_matrix.values)
        self.explained_variance = svd.explained_variance_ratio_.sum()
        print(f"svd: {self.ea_matrix.shape[0]} x {self.ea_matrix.shape[1]} -> "
              f"{X.shape[1]} dims ({self.explained_variance:.1%} variance)")
        self.embeddings = {
            name: X[i] for i, name in enumerate(self.ea_matrix.index)
        }
        return svd


class SpectralGraphEmbedder(Embedder):
    """
    spectral embedding on the graph laplacian.
    deterministic, no training, captures global graph structure.

    graph: networkx graph
    dimensions: output embedding size
    """

    def __init__(self, graph, dimensions=512):
        self.graph = graph
        self.dimensions = dimensions
        super().__init__()

    def method(self):
        nodes = list(self.graph.nodes())
        n = len(nodes)
        dims = min(self.dimensions, n - 2)

        A = nx.adjacency_matrix(self.graph, nodelist=nodes, weight="weight")
        se = SklearnSpectral(n_components=dims, affinity="precomputed",
                             random_state=42)
        X = se.fit_transform(A.toarray())

        print(f"spectral: {n} nodes -> {dims} dims")
        self.embeddings = {nodes[i]: X[i] for i in range(n)}
        return se


class LINEEmbedder(Embedder):
    """
    large-scale information network embedding.
    two objectives: first-order (direct neighbors) and
    second-order (shared neighbor structure). handles edge weights.

    graph: networkx graph
    dimensions: output embedding size (split half/half between orders)
    epochs: training epochs
    lr: learning rate
    neg_samples: negative samples per positive
    """

    def __init__(self, graph, dimensions=512, epochs=15, lr=0.005, neg_samples=5):
        self.graph = graph
        self.dimensions = dimensions
        self.epochs = epochs
        self.lr = lr
        self.neg_samples = neg_samples
        self.vocab = {}
        super().__init__()

    def method(self):
        nodes = list(self.graph.nodes())
        node_to_ix = {n: i for i, n in enumerate(nodes)}
        self.vocab = node_to_ix
        n = len(nodes)
        half_dim = self.dimensions // 2

        # build edge list with weights
        edges = []
        for u, v, d in self.graph.edges(data=True):
            w = d.get("weight", 1.0)
            edges.append((node_to_ix[u], node_to_ix[v], w))
            if not self.graph.is_directed():
                edges.append((node_to_ix[v], node_to_ix[u], w))

        if not edges:
            print("line: no edges")
            return None

        # degree distribution for negative sampling
        degrees = np.zeros(n)
        for u, v, w in edges:
            degrees[u] += w
            degrees[v] += w
        neg_dist = np.power(degrees, 0.75)
        neg_dist /= neg_dist.sum()

        print(f"line: {n} nodes, {len(edges)} edges")

        # first-order proximity
        emb_1 = self._train_order(edges, n, half_dim, neg_dist, order=1)
        # second-order proximity
        emb_2 = self._train_order(edges, n, half_dim, neg_dist, order=2)

        # concatenate both orders
        X = np.concatenate([emb_1, emb_2], axis=1)
        self.embeddings = {nodes[i]: X[i] for i in range(n)}
        return X

    def _train_order(self, edges, n, dim, neg_dist, order):
        emb = torch.nn.Embedding(n, dim)
        ctx = torch.nn.Embedding(n, dim)
        emb.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        ctx.weight.data.uniform_(-0.5 / dim, 0.5 / dim)

        optimizer = torch.optim.Adam(
            list(emb.parameters()) + list(ctx.parameters()), lr=self.lr
        )

        edge_arr = np.array(edges)
        weights = edge_arr[:, 2].astype(float)
        weight_dist = weights / weights.sum()

        batch_size = 1024
        n_batches = max(1, len(edges) // batch_size)

        for epoch in range(self.epochs):
            total_loss = 0
            # sample edges proportional to weight
            sampled = np.random.choice(len(edges), size=len(edges), p=weight_dist)

            for i in range(0, len(sampled), batch_size):
                batch_idx = sampled[i:i + batch_size]
                src = torch.LongTensor(edge_arr[batch_idx, 0].astype(int))
                dst = torch.LongTensor(edge_arr[batch_idx, 1].astype(int))
                neg = torch.LongTensor(
                    np.random.choice(n, size=(len(batch_idx), self.neg_samples),
                                     p=neg_dist)
                )

                optimizer.zero_grad()

                if order == 1:
                    # first-order: both use emb
                    pos_score = torch.nn.functional.logsigmoid(
                        torch.sum(emb(src) * emb(dst), dim=1))
                    neg_score = sum(
                        torch.nn.functional.logsigmoid(
                            -torch.sum(emb(src) * emb(neg[:, j]), dim=1))
                        for j in range(self.neg_samples)
                    )
                else:
                    # second-order: src uses emb, dst/neg use ctx
                    pos_score = torch.nn.functional.logsigmoid(
                        torch.sum(emb(src) * ctx(dst), dim=1))
                    neg_score = sum(
                        torch.nn.functional.logsigmoid(
                            -torch.sum(emb(src) * ctx(neg[:, j]), dim=1))
                        for j in range(self.neg_samples)
                    )

                loss = -torch.mean(pos_score + neg_score)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  line order-{order} epoch {epoch+1}/{self.epochs}, "
                  f"loss: {total_loss / n_batches:.4f}")

        with torch.no_grad():
            return emb.weight.numpy()


class PathwayMetapath2vec(Embedder):
    def __init__(self, graph, name, walks_per_node=10, walk_length=100):
        self.graph = graph
        self.name = name
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
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

    def train_embeddings(self, walks, dimensions=512, window_size=5, epochs=10,
                         lr=0.025, batch_size=1024, n_neg=5, walks_per_chunk=5000):
        """
        skip-gram with negative sampling. streams (target, context) pairs per
        chunk of walks instead of materializing every pair upfront, so memory
        stays O(chunk) rather than O(all pairs).
        """
        word_to_ix = {}
        for walk in walks:
            for word in walk:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        vocab_size = len(word_to_ix)
        self.vocab = word_to_ix

        # encode walks once as int arrays (reused across epochs)
        ix_walks = [
            np.fromiter((word_to_ix[w] for w in walk), dtype=np.int64, count=len(walk))
            for walk in walks
        ]

        # precompute context offsets ex. [-5,-4,-3,-2,-1,1,2,3,4,5]
        offsets = np.arange(-window_size, window_size + 1)
        offsets = offsets[offsets != 0]

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

        def _pairs_for_walk(walk):
            # vectorized window extraction, returns (targets, contexts) int arrays
            L = walk.shape[0]
            if L < 2:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
            pos = np.arange(L)
            ctx_pos = pos[:, None] + offsets[None, :]
            valid = (ctx_pos >= 0) & (ctx_pos < L)
            clipped = np.clip(ctx_pos, 0, L - 1)
            tgt = np.broadcast_to(walk[:, None], (L, offsets.size))[valid]
            ctx = walk[clipped][valid]
            return tgt, ctx

        def _train_batch(tgt_np, ctx_np):
            targets = torch.from_numpy(tgt_np)
            contexts = torch.from_numpy(ctx_np)
            neg = torch.from_numpy(
                np.random.randint(0, vocab_size, size=(tgt_np.shape[0], n_neg),
                                  dtype=np.int64)
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
            return loss.item()

        walk_order = np.arange(len(ix_walks))
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            np.random.shuffle(walk_order)

            # process walks in chunks to amortize vectorization
            for chunk_start in range(0, len(walk_order), walks_per_chunk):
                chunk_idx = walk_order[chunk_start:chunk_start + walks_per_chunk]
                tgts, ctxs = [], []
                for wi in chunk_idx:
                    t, c = _pairs_for_walk(ix_walks[wi])
                    if t.size:
                        tgts.append(t)
                        ctxs.append(c)
                if not tgts:
                    continue
                tgt_all = np.concatenate(tgts)
                ctx_all = np.concatenate(ctxs)

                # shuffle pairs within chunk, then iterate batches
                perm = np.random.permutation(tgt_all.shape[0])
                tgt_all = tgt_all[perm]
                ctx_all = ctx_all[perm]

                for i in range(0, tgt_all.shape[0], batch_size):
                    total_loss += _train_batch(
                        tgt_all[i:i + batch_size],
                        ctx_all[i:i + batch_size],
                    )
                    n_batches += 1

            print(f"epoch {epoch + 1}/{epochs}, loss: "
                  f"{total_loss / max(1, n_batches):.4f}, batches: {n_batches}")

        self.model = model
        with torch.no_grad():
            self.embeddings = {
                word: model.embeddings.weight[idx].numpy()
                for word, idx in word_to_ix.items()
            }
        return model

    def save_model(self, path):
        torch.save({"embeddings": self.embeddings, "vocab": self.vocab}, path)

    def load_model(self, path):
        state = torch.load(path)
        self.embeddings = state["embeddings"]
        self.vocab = state["vocab"]
