# NLP

## Smart Transliterator

Transliterates or returns correct spelling of original word unlike standard transliterators.

```python
transliterator = Transliterator('ua')

text = 'Садок вишневий коло хати, хрущі над вишнями гудуть. 1847'
transliterator.transliterate(text)
# 'Sadok vyshnevyy kolo khaty, khrushchi nad vyshnyamy gudut. 1847'

text = 'Юкрейніан Інновейшнз Компані'
transliterator.transliterate(text)
# 'Ukrainian Innovations Company'
```

Model implemented with Conditional Random Fields (CRF).

Supported languages: Russian, Ukrainian.

[Tutorial](../master/notebooks/transliterator.ipynb 'Transliterator Tutorial')


## Fuzzy Matcher

Hybrid fuzzy matcher based on Jaccard algorithm with modifications.

By default, Levenshtein is used as similarity function.

Modifications:

* Score is weighted by number of matched characters in word

```python
left = 'coca cola'
right = 'coca colas'
matcher.get_sim_score(left, right)
# 0.8941

left = 'sergeant pepper'
right = 'sergeant peppers'
matcher.get_sim_score(left, right)
# 0.936
```

* Word order matters or not

```python
left = 'coca cola company'
right = 'company coca cola'
matcher.get_sim_score(left, right, word_order_matters=True)
# 0.6667

left = 'coca cola company'
right = 'company coca cola'
matcher.get_sim_score(left, right, word_order_matters=False)
# 1.0
```

* Score is adjusted if one argument is substring of another

```python
left = 'coca cola'
right = 'welcome to coca cola company'
matcher.get_sim_score(left, right, substring=True)
# 1.0

left = 'coca colas'
right = 'welcome to coca cola company'
matcher.get_sim_score(left, right, substring=True)
# 0.8941

left = 'coca colas'
right = 'welcome to coca cola company'
matcher.get_sim_score(left, right, substring=False)
# 0.4606
```

[Tutorial](../master/notebooks/fuzzy_matcher.ipynb 'Fuzzy Matcher Tutorial')


## String Splitter

Splits text without spaces into human readable format using Zipf's law and dynamic programming.

```python
text = 'withalittlehelpfrommyfriends'
splitter.split(text)
# 'with a little help from my friends'

text = 'lucyintheskywithdiamonds'
splitter.split(text)
# 'lucy in the sky with diamonds'

text = 'whilemyguitargentlyweeps'
splitter.split(text)
# 'while my guitar gently weeps'
```

[Tutorial](../master/notebooks/string_splitter.ipynb 'String Splitter Tutorial')


## Named Entity Labeler (CRF)

Labels named entities in text. Implemented with Conditional Random Fields (CRF).

```python
text = 'Mr. Puigdemont has appeared in public in Brussels with several colleagues after declaring independence from Spain on October 27.'
labeler.predict(text)

# [('Mr', 'B-per'),
# ('Puigdemont', 'I-per'),
# ('has', 'O'),
# ('appeared', 'O'),
# ('in', 'O'),
# ('public', 'O'),
# ('in', 'O'),
# ('Brussels', 'B-geo'),
# ('with', 'O'),
# ('several', 'O'),
# ('colleagues', 'O'),
# ('after', 'O'),
# ('declaring', 'O'),
# ('independence', 'O'),
# ('from', 'O'),
# ('Spain', 'B-geo'),
# ('on', 'O'),
# ('October', 'B-tim'),
# ('27', 'I-tim')]
```

[Tutorial](../master/notebooks/ner_crf.ipynb 'NER CRF Tutorial')
