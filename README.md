# Probabilistic Lambek Categorial Sequent

This is the code for COLING2024 paper: A Generative Model For Lambek Categorial Sequents

Before executing the code, replace ./data/ with your custom sets. This repo only contains toy examples.

Example command for NLL:

```python parametrize_and_test.py --par_depth 2 ```

Example command for ranker, --max_len is upper bound of the sequent, longer sequent will be skipped:

```python prob_rank.py  --par_depth 2 --max_len 40 --input input.txt```

# Citation
```
@article{zhaoplcg,
  title={A Generative Model For Lambek Categorial Sequents}, 
  author={Jinman Zhao and Gerald Penn},
  booktitle = "Proceedings of the 2024 Joint International Conference on Computational
  Linguistics, Language Resources and Evaluation",
  month = may,
  year = "2024",
  address = "Turin, Italy",
}
```
