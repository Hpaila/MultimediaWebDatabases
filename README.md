# MultimediaWebDatabases
Group project for CSE 515 @ Arizona State University


To run:
```python code/task0a.py --gestures_dir data/ --window 3 --shift 3 --resolution 3```
While executing task0a.py, input directory containing the gestures is data/, if you want to run on sample data use "--gestures_dir sample/"

```python code/task0b.py --output_dir outputs/```

```python code/task1.py --output_dir outputs/ --vector_model tf --k 10 --user_option pca```

While executing task1.py, user_option can be pca, svd, nmf or lda

```python code/task2.py --query_word_file_name 2_words.csv --vector_model tf --output_dir outputs/ --query_output_dir outputs/ --user_option pca```

While executing task2.py, user_option can be dot_product, pca, svd, nmf, lda, edit_distance or dtw

``` python code/task3.py --vector_model tf --output_dir outputs/ --user_option pca --p 10 --type svd ```
While executing task3.py, user_option can be dot_product, pca, svd, nmf, lda, edit_distance or dtw.
type should be svd or nmf only. This is for dimensionality reduction on the similarity matrix

``` python code/task4.py --output_dir outputs/ --user_option pca --p 3 ```
While executing task4.py, user_option can be dot_product, pca, svd, nmf, lda, edit_distance or dtw.
p is the number of clusters that we want to form
