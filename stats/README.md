This directory contains:

- `metadata.csv`: a CSV file containing metadata about the speakers of The Androids Corpus
- `metadata_exploration.ipynb`: a notebook performing exploratory data analysis on the metadata of The Androids Corpus
- `build_master_dataset.py`: a python script to build a master dataset on with metadata and features extracted from the corpus
    - to run it, from the project root folder:
    ```python
    python build_master_dataset.py path/to/<input_dir> path/to/<output_dir>
   ```
   The resulting CSV will be located in the `output_dir`