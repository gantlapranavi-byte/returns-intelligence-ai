import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading reviews...")
reviews = pd.read_csv(f"{DATA_DIR}/olist_order_reviews_dataset.csv")
print(f"Total reviews: {len(reviews):,}")

reviews = reviews.dropna(subset=["review_comment_message"]).copy()
print(f"Reviews with text: {len(reviews):,}")

PT_STOPWORDS = set("""
a o e de da do em um uma para com nao que se na no os as dos das ao aos por foi
mas eu meu minha eles ele ela elas voce nos isso isto aquilo este esta esse essa
ja tao muito muita pouco pouca tambem mais menos sim sao tem ter
foi ser estar fazer fiz feito como quando onde porque pois porem entao
ate desde sobre entre antes depois agora hoje ontem amanha quem qual quais
todo toda todos todas algum alguma alguns algumas nada nenhum nenhuma cada outro
outra outros outras mesmo mesma mesmos mesmas tudo seu sua seus suas nosso nossa
nossos nossas vou vai vao vamos esta estou estao estava estavam estive
estiveram estivemos sou somos sao era eram fui foram foi fora fossem fomos
recebi recebeu recebido produto chegou veio comprei
""".split())

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zaaaaeeiooouuc ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokens(t):
    return [w for w in t.split() if len(w) > 2 and w not in PT_STOPWORDS]

reviews["clean_text"] = reviews["review_comment_message"].apply(clean_text)

neg = reviews[reviews["review_score"] <= 2].copy()
pos = reviews[reviews["review_score"] >= 4].copy()
print(f"\nNegative reviews (1-2 stars): {len(neg):,}")
print(f"Positive reviews (4-5 stars): {len(pos):,}")

def top_words(series, n=25):
    counter = Counter()
    for t in series:
        counter.update(tokens(t))
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])

neg_words = top_words(neg["clean_text"], n=25)
pos_words = top_words(pos["clean_text"], n=25)

print("\nTop 15 words in NEGATIVE reviews:")
print(neg_words.head(15).to_string(index=False))

print("\nTop 15 words in POSITIVE reviews:")
print(pos_words.head(15).to_string(index=False))

neg_words.to_csv(f"{OUT_DIR}/top_words_negative.csv", index=False)
pos_words.to_csv(f"{OUT_DIR}/top_words_positive.csv", index=False)

def make_wordcloud(text_series, title, filename, color):
    text = " ".join(" ".join(tokens(t)) for t in text_series)
    if not text.strip():
        return
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap=color,
        max_words=100,
        collocations=False,
    ).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18, pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")

make_wordcloud(neg["clean_text"], "Most Common Words - Negative Reviews (1-2 stars)",
               "wordcloud_negative.png", "Reds")
make_wordcloud(pos["clean_text"], "Most Common Words - Positive Reviews (4-5 stars)",
               "wordcloud_positive.png", "Greens")

print("\nClustering negative reviews into themes (KMeans on TF-IDF)...")

if len(neg) > 50:
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words=list(PT_STOPWORDS),
        ngram_range=(1, 2),
        min_df=5,
    )
    X = vectorizer.fit_transform(neg["clean_text"])

    n_clusters = 5
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    neg["cluster"] = km.fit_predict(X)

    feature_names = vectorizer.get_feature_names_out()
    print(f"\nFound {n_clusters} complaint themes:")
    cluster_summaries = []
    for i in range(n_clusters):
        center = km.cluster_centers_[i]
        top_idx = center.argsort()[-10:][::-1]
        top_terms = [feature_names[j] for j in top_idx]
        size = (neg["cluster"] == i).sum()
        print(f"  Theme {i+1}  (n={size}):  {', '.join(top_terms)}")
        cluster_summaries.append({
            "theme_id": i + 1,
            "size": int(size),
            "top_terms": ", ".join(top_terms),
        })

    pd.DataFrame(cluster_summaries).to_csv(f"{OUT_DIR}/complaint_themes.csv", index=False)
    print(f"\nSaved complaint themes to {OUT_DIR}/complaint_themes.csv")
else:
    print("Not enough negative reviews to cluster.")

plt.figure(figsize=(11, 7))
plt.barh(neg_words.head(20)["word"][::-1], neg_words.head(20)["count"][::-1], color="firebrick")
plt.title("Top 20 Words in Negative Reviews", fontsize=14)
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/top_negative_words_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nNLP analysis done.")