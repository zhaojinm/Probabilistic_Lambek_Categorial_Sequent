# Probabilistic Lambek Categorial Sequent

This is the code for COLING2024 paper: A Generative Model For Lambek Categorial Sequents

Before executing the code, replace ./data/ with your custom sets. This repo only contains toy examples.

Example command for NLL:

```python parametrize_and_test.py --par_depth 2 ```

Example command for ranker, --max_len is upper bound of the sequent, longer sequent will be skipped:

```python prob_rank.py  --par_depth 2 --max_len 40 --input input.txt```

# Citation
```
@inproceedings{zhao-penn-2024-generative-model,
    title = "A Generative Model for {L}ambek Categorial Sequents",
    author = "Zhao, Jinman  and
      Penn, Gerald",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.50",
    pages = "584--593",
    abstract = "In this work, we introduce a generative model, PLC+, for generating Lambek Categorial Grammar(LCG) sequents. We also introduce a simple method to numerically estimate the model{'}s parameters from an annotated corpus. Then we compare our model with probabilistic context-free grammars (PCFGs) and show that PLC+ simultaneously assigns a higher probability to a common corpus, and has greater coverage.",
}
```
