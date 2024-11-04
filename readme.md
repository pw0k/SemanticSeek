# SemanticSeek

**SemanticSeek** — система семантического поиска статей из набора данных **BBC News** с использованием FAISS и Sentence
Transformers.

## dataset

Dataset on BBC News Topic Classification consisting of 2,225 articles published on the BBC News
website corresponding during 2004-2005. Each article is labeled under one of 5 categories: business, entertainment,
politics, sport or tech.

[bbc-text.csv](https://huggingface.co/datasets/SetFit/bbc-news)

## before starting

`pip install pandas sentence-transformers faiss-cpu tqdm`

## using

`python main.py`

## examples query

`tv technology advancements`

```
--- Результат 1 ---
Категория: tech
Расстояние: 0.1234
Сниппет: tv future in the hands of viewers with home theatre systems plasma high-definition tvs ...

--- Результат 2 ---
Категория: business
Расстояние: 0.2345
Сниппет: worldcom boss left books alone former worldcom boss bernie ebbers who is accused ...

--- Результат 3 ---
Категория: entertainment
Расстояние: 0.3456
Сниппет: ocean s twelve raids box office ocean s twelve the crime caper sequel starring george ...

--- Результат 4 ---
Категория: politics
Расстояние: 0.4567
Сниппет: howard hits back at mongrel jibe michael howard has said a claim by peter hain ...

--- Результат 5 ---
Категория: sport
Расстояние: 0.5678
Сниппет: tigers wary of farrell gamble leicester say they will not be rushed into making a bid ...

```

где расстояние — это метрика, используемая для измерения схожести между эмбеддингами запроса и статей. 
В данном проекте используется L2 расстояние (евклидово расстояние).

Что Такое Расстояние:

Чем меньше расстояние между двумя векторами, тем более схожи они по смыслу.
L2 расстояние рассчитывается как квадратный корень из суммы квадратов разностей соответствующих компонентов векторов.
На Что Влияет Расстояние:

Релевантность Результатов: Меньшие значения расстояния означают более релевантные результаты.
Сортировка: Результаты сортируются по возрастанию расстояния, начиная с наиболее похожих статей.
Примеры Хороших Запросов
Запрос: "financial fraud in telecommunications"

Описание: Найдет статьи, связанные с бизнесом и мошенничеством в телекоммуникационной сфере.
Запрос: "advancements in high-definition television"

Описание: Фокусируется на технологических улучшениях в области высококачественного телевидения.
