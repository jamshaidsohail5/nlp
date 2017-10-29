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

Hybrid fuzzy matcher based on Jaccard algorithm with modifications. By default, as similarity function Levenshtein is used.

Modifications:

* Score is weighted by number of matched characters in word

```python
left = 'coca cola'
rigth = 'coca colas'
matcher.get_sim_score(left, rigth)
# 0.8941

left = 'sergeant pepper'
rigth = 'sergeant peppers'
matcher.get_sim_score(left, rigth)
# 0.936
```

* Word order matters or not

```python
left = 'coca cola company'
rigth = 'company coca cola'
matcher.get_sim_score(left, rigth, word_order_matters=True)
# 0.6667

left = 'coca cola company'
rigth = 'company coca cola'
matcher.get_sim_score(left, rigth, word_order_matters=False)
# 1.0
```

* Score is adjusted if one argument is substring of another

```python
left = 'coca cola'
rigth = 'welcome to coca cola company'
matcher.get_sim_score(left, rigth, substring=True)
# 1.0

left = 'coca colas'
rigth = 'welcome to coca cola company'
matcher.get_sim_score(left, rigth, substring=True)
# 0.8941

left = 'coca colas'
rigth = 'welcome to coca cola company'
matcher.get_sim_score(left, rigth, substring=False)
# 0.4606
```

[Tutorial](../master/notebooks/fuzzy_matcher.ipynb 'Fuzzy Matcher Tutorial')