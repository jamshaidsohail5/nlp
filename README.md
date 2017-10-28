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

[Tutorial](../master/notebooks/transliterator.ipynb 'Tutorial')
