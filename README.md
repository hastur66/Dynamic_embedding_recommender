# Dynamic embedding recommender

Modern recommenders heavily rely on embeddings to create vector representations of users and candidate items. These embeddings are crucial for calculating similarity between users and items, enabling the recommendation of relevant items to users.

* Embeddings in Recommenders: Modern recommenders utilize embeddings to represent users and items as vectors.

* Scale Challenge: As datasets grow, storing embedding tables in memory becomes impractical, especially with millions or billions of items.

* Sparse Data: Many items may be rarely seen, making dedicated embeddings inefficient.

* Dynamic Embedding Tables:
    - Represent infrequently occurring items with shared embeddings.
    - Significantly reduce the size of embedding tables.
    - Incur minimal performance costs.

* Benefits:
    - Efficient handling of large-scale data.
    - Effective recommendation generation while mitigating memory constraints.

[Reference](https://blog.tensorflow.org/2023/04/training-recommendation-model-with-dynamic-embeddings.html?linkId=9286461)
