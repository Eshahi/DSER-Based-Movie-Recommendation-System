##############################
# precompute_dser.py
##############################
import os
import random
import ast
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# -------------- DSER Model Classes --------------
class GMF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=8):
        super(GMF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

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
        out = self.mlp(x)
        return out

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
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, user_idx, item_idx,
                user_docvec, item_wordvec, item_contentvec):
        gmf_vec = self.gmf(user_idx, item_idx)
        mlp_vec = self.mlp(user_docvec, item_wordvec, item_contentvec)
        concat = torch.cat([gmf_vec, mlp_vec], dim=1)
        logit = self.out(concat).squeeze(-1)
        return torch.sigmoid(logit)

# -------------- Utility Functions --------------
def parse_json(json_str):
    try:
        return ast.literal_eval(json_str) if pd.notnull(json_str) else None
    except ValueError:
        return None

def list_to_str(lst):
    if not isinstance(lst, list):
        return ""
    return " ".join(str(x) for x in lst)

def safe_str(x):
    return x if pd.notnull(x) else ""

def load_data():
    # Example paths -- adapt as needed
    ratings_file = "data/ratings_small.csv"
    movies_file  = "movies_metadata_preprocessed.csv"
    keywords_file= "cleaned_movie_keywords.csv"
    credits_file = "cleaned_movie_credits.csv"

    ratings_df = pd.read_csv(ratings_file)
    movies_df  = pd.read_csv(movies_file)
    keywords_df= pd.read_csv(keywords_file)
    credits_df = pd.read_csv(credits_file)

    # rename 'id'->'movieId' if needed
    keywords_df.rename(columns={'id':'movieId'}, inplace=True)
    credits_df.rename(columns={'id':'movieId'}, inplace=True)
    movies_df.rename(columns={'id':'movieId'}, inplace=True)

    # convert rating >=3 => 1
    ratings_df['implicit'] = (ratings_df['rating'] >= 3).astype(int)
    # sort
    ratings_df.sort_values(by=['userId','timestamp'], inplace=True)
    ratings_df.reset_index(drop=True, inplace=True)

    # merge for content
    movie_content_df = movies_df.merge(keywords_df, on='movieId', how='left')
    movie_content_df = movie_content_df.merge(credits_df, on='movieId', how='left')

    # build id2title
    id2title = dict(zip(movies_df['movieId'], movies_df['title']))

    return ratings_df, movie_content_df, id2title

def build_doc2vec_embeddings(ratings_df, movie_content_df,
                             user_d2v_size=40, user_d2v_window=10,
                             content_d2v_size=40):
    # Build user sequences
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()

    user_sequences = {}
    for row in ratings_df.itertuples(index=False):
        u = row.userId
        i = row.movieId
        user_sequences.setdefault(u, []).append(str(i))

    # train doc2vec for user+item sequences
    tagged_docs = []
    for u, items in user_sequences.items():
        tagged_docs.append(TaggedDocument(words=items, tags=[str(u)]))

    user_item_model = Doc2Vec(
        documents=tagged_docs,
        vector_size=user_d2v_size,
        window=user_d2v_window,
        min_count=1,
        dm=0,
        epochs=10,
        workers=4
    )

    user_docvec_map = {}
    for u in unique_users:
        user_docvec_map[u] = user_item_model.dv[str(u)]

    item_wordvec_map = {}
    for m in unique_items:
        if str(m) in user_item_model.wv:
            item_wordvec_map[m] = user_item_model.wv[str(m)]
        else:
            item_wordvec_map[m] = np.zeros(user_d2v_size, dtype=np.float32)

    # doc2vec for item content
    item_content_docs = []
    for row in movie_content_df.itertuples(index=False):
        mid = row.movieId
        overview_text = safe_str(row.overview)

        kw_text   = list_to_str(parse_json(row.keywords)) if isinstance(row.keywords, str) else list_to_str(row.keywords)
        cast_text = list_to_str(parse_json(row.cast))     if isinstance(row.cast, str) else list_to_str(row.cast)
        crew_text = list_to_str(parse_json(row.crew))     if isinstance(row.crew, str) else list_to_str(row.crew)

        combined = overview_text + " " + kw_text + " " + cast_text + " " + crew_text
        item_content_docs.append(
            TaggedDocument(words=combined.lower().split(), tags=[f"ITEM_{int(mid)}"])
        )

    content_model = Doc2Vec(
        documents=item_content_docs,
        vector_size=content_d2v_size,
        window=10,
        min_count=1,
        dm=0,
        epochs=10,
        workers=4
    )

    item_content_map = {}
    for row in movie_content_df.itertuples(index=False):
        mid = row.movieId
        tag_id = f"ITEM_{int(mid)}"
        if tag_id in content_model.dv:
            item_content_map[mid] = content_model.dv[tag_id]
        else:
            item_content_map[mid] = np.zeros(content_d2v_size, dtype=np.float32)

    return user_docvec_map, item_wordvec_map, item_content_map

def train_dser_model(ratings_df,
                     user_docvec_map, item_wordvec_map, item_content_map,
                     user_d2v_size=40, content_d2v_size=40,
                     EPOCHS=3, BATCH_SIZE=512, lr=0.01):
    # Build indexes
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()
    user2index   = {u:i for i,u in enumerate(unique_users)}
    item2index   = {m:i for i,m in enumerate(unique_items)}

    num_users = len(unique_users)
    num_items = len(unique_items)

    # negative sampling
    all_data = ratings_df[['userId','movieId','implicit']].drop_duplicates()
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(0.8*len(all_data))
    train_df  = all_data.iloc[:split_idx]

    all_ui_set = set(zip(all_data['userId'], all_data['movieId']))
    train_pos  = train_df[train_df['implicit'] == 1]

    train_instances = []
    neg_ratio = 4
    for row in train_pos.itertuples(index=False):
        u, m = row.userId, row.movieId
        train_instances.append((u, m, 1))
        for _ in range(neg_ratio):
            neg_item = random.choice(unique_items)
            while (u, neg_item) in all_ui_set:
                neg_item = random.choice(unique_items)
            train_instances.append((u, neg_item, 0))

    train_np = np.array(train_instances, dtype=np.int64)

    # device
    device = torch_directml.device(0)

    # DSER setup
    gmf_emb_size=8
    mlp_input_dim = user_d2v_size + user_d2v_size + content_d2v_size
    mlp_layers=[64,32,16]
    model = DSER(num_users, num_items,
                 gmf_emb_size=gmf_emb_size,
                 mlp_input_dim=mlp_input_dim,
                 mlp_layers=mlp_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion= nn.BCELoss()

    for ep in range(EPOCHS):
        np.random.shuffle(train_np)
        epoch_loss = 0.0
        model.train()

        for start in range(0, len(train_np), BATCH_SIZE):
            end = min(start+BATCH_SIZE, len(train_np))
            batch = train_np[start:end]

            user_list = batch[:,0]
            item_list = batch[:,1]
            label_list= batch[:,2]

            user_idx = torch.LongTensor([user2index[u] for u in user_list]).to(device)
            item_idx = torch.LongTensor([item2index[i] for i in item_list]).to(device)
            y_batch  = torch.FloatTensor(label_list).to(device)

            default_content_vec = np.zeros(content_d2v_size, dtype=np.float32)
            user_vecs, item_seq_vecs, item_cnt_vecs = [], [], []
            for (u,i) in zip(user_list, item_list):
                user_vecs.append(user_docvec_map[u].copy())
                item_seq_vecs.append(item_wordvec_map[i].copy())
                if i in item_content_map:
                    item_cnt_vecs.append(item_content_map[i].copy())
                else:
                    item_cnt_vecs.append(default_content_vec)

            user_vec_t = torch.FloatTensor(user_vecs).to(device)
            item_seq_t= torch.FloatTensor(item_seq_vecs).to(device)
            item_cnt_t= torch.FloatTensor(item_cnt_vecs).to(device)

            optimizer.zero_grad()
            preds = model(user_idx, item_idx, user_vec_t, item_seq_t, item_cnt_t)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {ep+1}/{EPOCHS}, Loss={epoch_loss:.4f}")

    return model, user2index, item2index, device, num_users, num_items


def main():
    print("Loading data and building doc2vec embeddings...")
    ratings_df, movie_content_df, id2title = load_data()
    user_docvec_map, item_wordvec_map, item_content_map = build_doc2vec_embeddings(
        ratings_df, movie_content_df,
        user_d2v_size=40, user_d2v_window=10,
        content_d2v_size=40
    )
    print("Embeddings built.")

    print("Training DSER model offline...")
    model, user2index, item2index, device, num_users, num_items = train_dser_model(
        ratings_df,
        user_docvec_map, item_wordvec_map, item_content_map,
        user_d2v_size=40, content_d2v_size=40,
        EPOCHS=3, BATCH_SIZE=512, lr=0.01
    )
    print("Training done.")

    # ----- Save artifacts -----
    print("Saving artifacts to disk...")

    # 1) Save dictionaries
    with open("user_docvec_map.pkl","wb") as f:
        pickle.dump(user_docvec_map, f)

    with open("item_wordvec_map.pkl","wb") as f:
        pickle.dump(item_wordvec_map, f)

    with open("item_content_map.pkl","wb") as f:
        pickle.dump(item_content_map, f)

    with open("user2index.pkl","wb") as f:
        pickle.dump(user2index, f)

    with open("item2index.pkl","wb") as f:
        pickle.dump(item2index, f)

    with open("id2title.pkl","wb") as f:
        pickle.dump(id2title, f)

    # 2) Save DSER model weights
    torch.save(model.state_dict(), "dser_model_weights.pth")
    print("All precomputed data saved. Done.")

if __name__ == "__main__":
    main()
