
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_directml
import networkx as nx
import matplotlib.pyplot as plt
import os


class GMF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=8):
        super(GMF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        return u * i

class MLP(nn.Module):
    def __init__(self, input_dim=120, layers=[64,32,16]):
        super(MLP, self).__init__()
        seq = []
        prev_dim = input_dim
        for layer_size in layers:
            seq.append(nn.Linear(prev_dim, layer_size))
            seq.append(nn.ReLU())
            prev_dim = layer_size
        self.mlp = nn.Sequential(*seq)

    def forward(self, user_doc, item_word, item_content):
        x = torch.cat((user_doc, item_word, item_content), dim=1)
        return self.mlp(x)

class DSER(nn.Module):
    def __init__(self, num_users, num_items,
                 gmf_emb_size=8,
                 mlp_input_dim=120,
                 mlp_layers=[64,32,16]):
        super(DSER, self).__init__()
        self.gmf = GMF(num_users, num_items, emb_size=gmf_emb_size)
        self.mlp = MLP(input_dim=mlp_input_dim, layers=mlp_layers)
        final_dim = gmf_emb_size + mlp_layers[-1]
        self.out = nn.Linear(final_dim, 1)

    def forward(self, user_idx, item_idx,
                user_docvec, item_wordvec, item_contentvec):
        gmf_vec = self.gmf(user_idx, item_idx)
        mlp_vec = self.mlp(user_docvec, item_wordvec, item_contentvec)
        concat = torch.cat([gmf_vec, mlp_vec], dim=1)
        logit = self.out(concat).squeeze(-1)
        return torch.sigmoid(logit)



def visualize_dser_architecture():
    G = nx.DiGraph()

    # Add nodes
    G.add_node("UserIdx\n(Embedding)")
    G.add_node("ItemIdx\n(Embedding)")
    G.add_node("GMF\n(element-wise product)")

    G.add_node("UserDocVec\n(40-d)")
    G.add_node("ItemDocVec\n(40-d)")
    G.add_node("ItemContentVec\n(40-d)")
    G.add_node("MLP\n[120->64->32->16]")

    G.add_node("Concat\n(GMF+MLP)")
    G.add_node("Sigmoid\nOutput")

    # Add edges
    G.add_edge("UserIdx\n(Embedding)", "GMF\n(element-wise product)")
    G.add_edge("ItemIdx\n(Embedding)", "GMF\n(element-wise product)")

    G.add_edge("UserDocVec\n(40-d)", "MLP\n[120->64->32->16]")
    G.add_edge("ItemDocVec\n(40-d)", "MLP\n[120->64->32->16]")
    G.add_edge("ItemContentVec\n(40-d)", "MLP\n[120->64->32->16]")

    G.add_edge("GMF\n(element-wise product)", "Concat\n(GMF+MLP)")
    G.add_edge("MLP\n[120->64->32->16]", "Concat\n(GMF+MLP)")
    G.add_edge("Concat\n(GMF+MLP)", "Sigmoid\nOutput")

    fig, ax = plt.subplots(figsize=(8,6))
    pos = {
        "UserIdx\n(Embedding)": (0, 2),
        "ItemIdx\n(Embedding)": (0, 0),
        "GMF\n(element-wise product)": (1.5, 1),
        "UserDocVec\n(40-d)": (0,4),
        "ItemDocVec\n(40-d)": (0,3),
        "ItemContentVec\n(40-d)": (0,2.3),
        "MLP\n[120->64->32->16]": (1.5, 3.2),
        "Concat\n(GMF+MLP)": (3, 2),
        "Sigmoid\nOutput": (4.5, 2),
    }

    nx.draw_networkx(
        G, pos, ax=ax, arrows=True, with_labels=True,
        node_size=1800, node_color="#D0E6F5", font_size=8
    )
    plt.axis("off")
    return fig



def recommend_for_user(model, user_id,
                       user2index, item2index,
                       user_docvec_map, item_wordvec_map, item_content_map,
                       id2title,
                       topK=5, device=None):

    if user_id not in user_docvec_map:
        return []

    model.eval()
    if device is None:
        device = torch_directml.device(0)

    user_idx = user2index[user_id]
    user_doc = user_docvec_map[user_id].copy()
    user_doc_t = torch.FloatTensor(user_doc).unsqueeze(0).to(device)

    default_content_vec = np.zeros(item_content_map[next(iter(item_content_map))].shape[0], dtype=np.float32)

    all_items = list(item2index.keys())
    scores = []
    for it in all_items:
        it_idx = item2index[it]
        seq_vec = item_wordvec_map[it].copy()
        if it in item_content_map:
            cnt_vec = item_content_map[it].copy()
        else:
            cnt_vec = default_content_vec

        u_batch = torch.LongTensor([user_idx]).to(device)
        i_batch = torch.LongTensor([it_idx]).to(device)
        seq_t   = torch.FloatTensor(seq_vec).unsqueeze(0).to(device)
        cnt_t   = torch.FloatTensor(cnt_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            score = model(u_batch, i_batch, user_doc_t, seq_t, cnt_t).item()

        scores.append((it, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:topK]

    annotated = []
    for (mid, sc) in top_results:
        movie_title = id2title.get(mid, f"Unknown Title (ID={mid})")
        annotated.append((mid, movie_title, sc))

    return annotated



def main():
    st.title("DSER Recommender (Precomputed) With Graph & Ratings")

    st.write("Loading precomputed embeddings and model weights...")

    import pickle
    with open("user_docvec_map.pkl","rb") as f:
        user_docvec_map = pickle.load(f)
    with open("item_wordvec_map.pkl","rb") as f:
        item_wordvec_map = pickle.load(f)
    with open("item_content_map.pkl","rb") as f:
        item_content_map = pickle.load(f)
    with open("user2index.pkl","rb") as f:
        user2index = pickle.load(f)
    with open("item2index.pkl","rb") as f:
        item2index = pickle.load(f)
    with open("id2title.pkl","rb") as f:
        id2title = pickle.load(f)


    ratings_df = pd.read_csv("data/ratings_small.csv")

    num_users = len(user2index)
    num_items = len(item2index)
    mlp_input_dim = 40 + 40 + 40

    model = DSER(num_users, num_items,
                 gmf_emb_size=8,
                 mlp_input_dim=mlp_input_dim,
                 mlp_layers=[64,32,16])
    model.load_state_dict(torch.load("dser_model_weights.pth", map_location="cpu"))

    device = torch_directml.device(0)
    model.to(device)
    st.success("Artifacts loaded successfully.")

    with st.expander("Show DSER Network Architecture"):
        fig_arch = visualize_dser_architecture()
        st.pyplot(fig_arch)

    user_ids = sorted(user_docvec_map.keys())
    chosen_user = st.selectbox("Select user ID:", user_ids)

    st.subheader("5 Movies User Rated")
    user_ratings = ratings_df[ratings_df["userId"] == chosen_user].head(5)
    if user_ratings.empty:
        st.write("This user has no rated movies in the CSV.")
    else:
        for row in user_ratings.itertuples():
            mid = row.movieId
            rtg = row.rating
            ttl = id2title.get(mid, f"Unknown Title (ID={mid})")
            st.write(f"- MovieID={mid}, Title='{ttl}', Rating={rtg}")

    if st.button("Recommend Top-5"):
        st.write("Generating top-5 recommendations for user:", chosen_user)
        recs = recommend_for_user(
            model, chosen_user,
            user2index, item2index,
            user_docvec_map, item_wordvec_map, item_content_map,
            id2title,
            topK=5, device=device
        )
        if not recs:
            st.write("No recommendations. Possibly user not in docvec map.")
        else:
            st.write("Here are the recommended movies:")
            for (mid, title, score) in recs:
                st.write(f"MovieID={mid}, Title='{title}', Score={score:.4f}")


if __name__ == "__main__":
    main()
