# GEN-min-script
Repository with the models used for the task of segmentation and classification of legal documents.

## Setup enviroment
```
conda env create -f env.yml
conda activate gen-env-min
pip install -r requirements.txt
```

## Pre-annotation data analysis

The distribution of pages per document has been studied. Determining that the vast majority of PDFs contain between 1 and 50 pages. As a single page does not provide information for the task, files of more than 15 pages and less than 50 pages will be taken.

![Distribution of pages](EDA/graficas/distribucion_pag.png)

The duration of contracts has also been studied. It can be seen that at some point most of the contracts that ended before the year 2000 were deleted. On the other hand, it can be seen that most of the terminated contracts have a short duration.

![Start vs end](EDA/graficas/inicio_vs_fin.png)

Finally, it has been studied how many contracts have ended each year, with respect to the total we have.It should be noted that the contracts still open represent 71% of the data.

![Contracts by year](EDA/graficas/output.png)