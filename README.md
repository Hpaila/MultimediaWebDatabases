# MultimediaWebDatabases
Group project for CSE 515 @ Arizona State University


To run:
```python phase2/task0a.py --gestures_dir data/ --window 3 --shift 3 --resolution 3```
While executing task0a.py, input directory containing the gestures is data/, if you want to run on sample data use "--gestures_dir sample/"

```python phase2/task0b.py --output_dir outputs/```

```python phase2/task1.py --output_dir outputs/ --vector_model tf --k 10 --user_option pca```

While executing task1.py, user_option can be pca, svd, nmf or lda

```python phase2/task2.py --query_word_file_name 2_words.csv --vector_model tf --output_dir outputs/ --query_output_dir outputs/ --user_option pca```

While executing task2.py, user_option can be dot_product, pca, svd, nmf, lda, edit_distance or dtw
For edit_distance, use --custom_cost=true to enable the custom cost

``` python phase2/task3.py --vector_model tf --output_dir outputs/ --user_option pca --p 10 --type svd ```
While executing task3.py, user_option can be dot_product, pca, svd, nmf, lda, edit_distance or dtw.
type should be svd or nmf only. This is for dimensionality reduction on the similarity matrix

``` python phase2/task4.py --output_dir outputs/ --user_option pca --p 3 ```
While executing task4.py, user_option can be dot_product, pca, svd, nmf, lda, edit_distance or dtw.
p is the number of clusters that we want to form

Phase 3 commands:
Task 1

python phase3/task1.py --gestures_dir ./Phase3_data_for_report/ --k 5 --m 10 --n 1 2_7 31 256_4

--gestures_dir # represents the folder path for input gestures
--k # represents the number of outgoing edges for each node in adjacency graph
--m # number of dominant gestures
--n # list of user specified gestures

Optional params include:
--user_option pca # other options include svd, nmf, lda, dtw, edit_distance
--vector_model tf # other options include tf_idf
--user_option_k 10 # represents the number of dimensions to be reduced to
--output_dir ./outputs # represents the output directory to which the files get saved to
--window 3 --shift 3 --resolution 3 # represents the params to construct words from gestures

Task 2

Python phase3/task2.py --query_file 17.csv --nn 25 --gestures_dir ../Phase3_data_for_report/

--query_file # represents the query file which is to be classified
--nn # number of nearest neighbors
--gestures_dir # path for gestures directory

Optional parameters include:
--user_option pca # other options include svd, nmf, lda, dtw, edit_distance
--vector_model tf # other options include tf_idf
--user_option_k 10 # represents the number of dimensions to be reduced to
--output_dir ./outputs # represents the output directory to which the files get saved to
--window 3 --shift 3 --resolution 3 # represents the params to construct words from gestures

Task 3
python phase3/task3.py --k 6 --l 3 --vectors /vectors_path

Once the LSH index structure is generated, the console will ask to enter a query gesture to search, and the user needs to enter the query file name without .csv. For instance, if the user wants to search for 560.csv, then we should enter 560. Users can search for as many gestures as needed using the same index structure.

Task 4
python phase3/task4.py --query_gesture 560 --t 10

Initial search results using task3 will be displayed to the user in the console, then the console will ask to enter the relevant gestures first. User needs to enter the gesture file names without .csv separated by space. Next, the console will ask for non relevant gestures, User needs to enter the gesture file names without .csv separated by space. 
Then the updated results will be displayed to the user. Also, the feedback term weights are saved in a file called task4_output in phase3 directory.

Task 5
python phase3/task5.py --query_gesture 254 --t 10
“t” dominant features will be displayed to the user in the console, then the console will ask to enter the relevant gestures first. User needs to enter the gesture file names without .csv separated by space. Next, the console will ask for non relevant gestures, User needs to enter the gesture file names without .csv separated by space. 
Then the updated results will be displayed to the user.

Task 6
python phase3/task6.py
A window will open asking the user to enter the query gesture, the number of results to be returned and the feedback type probabilistic or ppr. Users can use this to search and provide relevance feedback and get the updated results.
