# SemanticSeek

**SemanticSeek** — семантический поиска статей из набора данных **BBC News** с использованием FAISS и Sentence
Transformers.

## dataset

Dataset on BBC News Topic Classification consisting of 2,225 articles published on the BBC News
website corresponding during 2004-2005. Each article is labeled under one of 5 categories: business, entertainment,
politics, sport or tech.

[bbc-text.csv](https://huggingface.co/datasets/SetFit/bbc-news)

### В данном проекте используется косинусная схожесть для измерения семантической близости между эмбеддингами запроса и статей.

Что Такое Косинусная Схожесть:
Семантическая Близость: Даже если два текста различаются по длине или количеству слов, они могут быть семантически похожими.
Косинусная схожесть эффективно захватывает эту близость.

Фокус на Направлении Векторов: Косинусная схожесть измеряет угол между двумя векторами, игнорируя их длину. Это позволяет
оценивать семантическую близость независимо от масштаба эмбеддингов.

## before starting

`pip install pandas sentence-transformers faiss-cpu tqdm`

## using

`python main.py`

## examples query

1.`technology advancements`

```
--- Результат 1 ---
Категория: tech
Схожесть: 0.6079
Сниппет:
when invention turns to innovation it is unlikely that future technological inventions are going to have the same kind of transformative impact that
they did in the past.  when history takes a look back at great inventions like the car and transistor  they were defining technologies which
ultimately changed people s lives substantially. but  says nick donofrio  senior vice-president of technology and manufacturing at ibm  it was not
the thing  itself that actually improved people s lives. it was all the social and cultural changes that the discovery or invention brought with it.
the car brought about a crucial change to how people lived in cities  giving them the ability to move out into the suburbs  whilst having mobility and
access.  when we talk about innovation and creating real value in the 21st century  we have to think more like this  but faster   mr donofrio told the
bbc news website  after giving the royal academy of engineering 2004 hinton lecture.  the invention  discovery is likely not to have the same value as
...
```
"technology advancements" ищем технологические достижения - статья о различие между изобретением и инновацией.


2. `financial offence`
```
--- Результат 1 ---
Категория: tech
Схожесть: 0.4422
Сниппет:
who do you think you are  the real danger is not what happens to your data as it crosses the net  argues analyst bill thompson. it is what happens
when it arrives at the other end.  the financial services authority has warned banks and other financial institutions that members of criminal gangs
may be applying for jobs which give them access to confidential customer data. the fear is not that they will steal money from our bank accounts but
that they will instead steal something far more valuable in our digital society - our identities. armed with the personal details that a bank holds
...
```
"financial offence" финансовые преступления - cтатья про угрозу кражи личных данных через сотрудников финансовых учреждений


