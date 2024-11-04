import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap

# Загрузка данных из CSV
def load_bbc_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'category' not in df.columns or 'text' not in df.columns:
            raise ValueError("CSV-файл должен содержать колонки 'category' и 'text'.")
        categories = df['category'].tolist()
        texts = df['text'].tolist()
        return categories, texts
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Ошибка: Файл '{file_path}' пуст.")
        exit(1)
    except pd.errors.ParserError:
        print(f"Ошибка: Файл '{file_path}' содержит некорректный формат.")
        exit(1)
    except Exception as e:
        print(f"Произошла ошибка при загрузке CSV: {e}")
        exit(1)

# Вычисление эмбеддингов с помощью Sentence Transformers
def compute_embeddings(model_name, texts, batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    # FAISS требует float32
    embeddings = np.array(embeddings).astype('float32')
    # Нормализация до единичной длины для косинусной схожести
    faiss.normalize_L2(embeddings)
    return embeddings, model

#  Индексация эмбеддингов с использованием FAISS
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    # Индекс на основе L2 расстояния
    # index = faiss.IndexFlatL2(dimension)
    # Индекс для внутреннего произведения
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

#  Функция для поиска похожих статей
def search_similar_articles(query, model, index, categories, texts, top_k=5):
    query_embedding = model.encode([query]).astype('float32')
    # Нормализация запроса
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            'Rank': i + 1,
            'Category': categories[idx],
            'Similarity': distances[0][i],
            'Snippet': texts[idx]
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

    print("SemanticSeek готов к работе!")

    while True:
        query = input("\nВведите запрос для поиска похожих статей (или 'exit' для выхода): ").strip()
        if query.lower() == 'exit':
            print("Завершение программы.")
            break
        if not query:
            print("Пожалуйста, введите непустой запрос.")
            continue

        print(f"\nПоиск похожих статей для запроса: \"{query}\"")
        results = search_similar_articles(query, model, index, categories, texts, top_k)

        for res in results:
            wrapped_snippet = textwrap.fill(res['Snippet'], width=150)
            print(f"\n--- Результат {res['Rank']} ---")
            print(f"Категория: {res['Category']}")
            print(f"Схожесть: {res['Similarity']:.4f}")
            print(f"Сниппет:\n{wrapped_snippet}")

    print("Завершение программы.")


if __name__ == "__main__":
    main()