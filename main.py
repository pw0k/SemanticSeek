import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# Загрузка данных из CSV
def load_bbc_csv(file_path):
    df = pd.read_csv(file_path)
    # Проверяем наличие необходимых колонок
    if 'category' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV-файл должен содержать колонки 'category' и 'text'.")
    categories = df['category'].tolist()
    texts = df['text'].tolist()
    return categories, texts


# Вычисление эмбеддингов с помощью Sentence Transformers
def compute_embeddings(model_name, texts, batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    # FAISS требует float32
    embeddings = np.array(embeddings).astype('float32')
    return embeddings, model

#  Индексация эмбеддингов с использованием FAISS
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Индекс на основе L2 расстояния
    index.add(embeddings)
    return index


#  Функция для поиска похожих статей
def search_similar_articles(query, model, index, categories, texts, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            'Rank': i + 1,
            'Category': categories[idx],
            'Snippet': texts[idx][:500] + "..." if len(texts[idx]) > 500 else texts[idx],
            'Distance': distances[0][i]
        }
        results.append(result)
    return results


def main():
    csv_file = 'bbc-news.csv'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    #выводим топ 5 статей
    top_k = 5

    print("Загрузка данных из CSV...")
    categories, texts = load_bbc_csv(csv_file)
    print(f"Загружено {len(texts)} документов.")

    print("Вычисление эмбеддингов...")
    embeddings, model = compute_embeddings(model_name, texts)

    print("Построение индекса FAISS...")
    index = build_faiss_index(embeddings)
    print(f"Индекс построен. Количество векторов в индексе: {index.ntotal}")

    while True:
        print("\nВведите запрос для поиска похожих статей (или 'exit' для выхода):")
        query = input().strip()
        if query.lower() == 'exit':
            break
        if not query:
            print("Пожалуйста, введите непустой запрос.")
            continue

        print(f"\nПоиск похожих статей для запроса: \"{query}\"")
        results = search_similar_articles(query, model, index, categories, texts, top_k)

        for res in results:
            print(f"\n--- Результат {res['Rank']} ---")
            print(f"Категория: {res['Category']}")
            print(f"Расстояние: {res['Distance']:.4f}")
            print(f"Сниппет: {res['Snippet']}")

    print("Завершение программы.")


if __name__ == "__main__":
    main()