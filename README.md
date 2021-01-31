Implementation of a paper in ICSME 2015
==============
This project implements the model ***TIE*** described in paper [Who Should Review This Change?](http://www.mysmu.edu/faculty/davidlo/papers/icsme15-review.pdf).

Files in this Repository
==============
- `retrieve_reviews.py` is a utility that retrieves reviews from Gerrit Systems.
- `tie_recommend.py` is main file that can be imported as a Python module.
- `train_and_test.py` is a utility that trains a model and tests it at the same time, and finally outputs recommendation results.
- `Android.json`, `LibreOffice.json`, `QT.json` and `OpenStack.json` are 4 example input JSON files.

How to use
==============

#### 1. retrieve_reviews.py

Run `retrieve_reviews.py` to retrieve historical reviews from Gerrit systems. You can modify the variable `projects` in this file.

Here I provide 4 JSON files that contains reviews from 4 big projects. You can use these files directly.

#### 2. train_and_test.py

Run `train_and_test.py` to train a recommendation model and test the model at the same time. The method is described in the paper:

> The process proceeds as follows: first, we train a *TIE* model by using the first review, and test the trained model by using the second review, then we update the *TIE* model by using the second review (with its ground truth reviewers). Next, we test using the third review, and update the *TIE* model by using the third review, and so on.

You should run the script with these parameters shown below:

| Parameter | Description |
| --- | --- |
| *reviews_file* | **Required**, the input JSON file containing the reviews. |
| *output_file* | **Required**, the file to output the results in. |
| *model_file* | **Required**, the file to store the model in. |
| *max_reviews* | **Optional**, the maximum number of reviews to be processed. This implementation will process all reviews if `max_reviews` is not specified. |
| *alpha* | Parameter `alpha` in original paper. |

Example command:

```bash
python train_and_test.py --reviews_file=Android.json --output_file=output/Android_output.json --model_file=output/Android_model.json
```

#### 3. tie_recommend.py

`tie_recommend.py` can be used as a Python module. Example usage is shown below:

```python
from tie_recommend import TIEModel
# some code omitted
# ...
tie_model = TIEModel(all_words, all_reviewers, alpha=0.7, M=50)
tie_model.update(reviews[0])
recomm_reviewers = tie_model.recommend(reviews[1], max_count=5)
print("Recommended Reviewers:", recomm_reviewers)
tie_model.save("temp.bin")
```

Input Format
==============
TIE model accepts an `dict` object as input review. Each review object contains fields `id`, `uploaded-time`, `textual-content` and `changed-files`:
- Field `id` is the unique identifier of the review.
- Field `uploaded-time` is the time when the review was committed to Gerrit system.
- Field `textual-content` is text content of the review, usually commit messages.
- Field `changed-files` is an array of changed files.

If the review object is used to **update the model**, then it must contain `reviewers` field.`reviewers` is an array of several reviewer object. Each reviewer object must at least contain field `id`(usually a 32-bit integer).

As for utility `train_and_test.py`, input JSON file contains an array of many review objects. Each review object **must** contain fields `id`, `uploaded-time`, `textual-content`, `changed-files` and `reviewers`.

An example input file is shown below:

```json
[
    {
        "id": "qt%2Fqtsensors~master~I1c0dd3dea4a0b296388f66fc44fed560da85e028",
        "uploaded-time": "2012-05-29 23:23:35",
        "reviewers": [
            {
                "id": 1000049,
                "name": "Qt Sanity Bot"
            },
            {
                "id": 1000097,
                "name": "Lincoln Ramsay"
            },
            {
                "id": 1000136,
                "name": "Lorn Potter"
            }
        ],
        "textual-content": "there is no main.qml file here. Now it shows up in creator.\n\nChange-Id: I1c0dd3dea4a0b296388f66fc44fed560da85e028\n",
        "changed-files": [
            "examples/QtSensors/QtSensors_accelbubble/QtSensors_accelbubble.pro"
        ]
    }
]
```

Output Format
==============
TIE model returns an array of recommended reviewers' IDs for each input review.

For utility `train_and_test.py`, output JSON object contains top-10 recommended reviewers' IDs for all reviews, together with statistics.

An example output file is shown below:

```json
{
    "top1-accuracy": 0.50,
    "top3-accuracy": 0.60,
    "top5-accuracy": 0.80,
    "top10-accuracy": 0.92,
    "mrr": 0.63,
    "recommendation-results": [
        {
            "review-id": "qt%2Fqtsensors~master~I1c0dd3dea4a0b296388e66fc44fed360da85e028",
            "result": [100003, 10013, 100014, 100055, 100023, 100011, 100033, 100004, 100008, 100009]
        },
        {
            "review-id": "qt-creator%2Fqt-creator~master~I5774d04a45f28a4e276a0ef283ce0aa5a2f2e553",
            "result": [100002, 10013, 100015, 100011, 100023, 100044, 100023, 100007, 100001, 100031]
        }
    ]
}
```

Results on 4 projects
==============

#### 1. Statistics of collected data

| Project | Time Period | Revi. | Re. | Avg. Re. |
| --- | --- | --- | --- | --- |
| Android | 2008/10 - 2012/01 | 8958 | 802 | 2.58 |
| LibreOffice | 2012/03 - 2014/06 | 8585 | 224 | 1.54 |
| OpenStack | 2011/07 - 2012/05 | 7064 | 367 | 3.97 |
| QT | 2011/05 - 2012/05 | 24734 | 530 | 3.31 |

Here `Revi.` refers to the number of collected reviews, `Re.` refers to the number of the unique reviewers extracted from collected reviews, and `Avg. Re.` refers to the average number of code reviewers per review.

#### 2. Results

| Project | Top-10 P.A. | Top-5 P.A. | Top-3 P.A. | Top-1 P.A. | MRR |
| --- | --- | --- | --- | --- | --- |
| Android | 0.95 | 0.90 | 0.84 | 0.65 | 0.76 |
| LibreOffice | 0.89 | 0.80 | 0.74 | 0.53 | 0.66 |
| OpenStack | 0.98 | 0.97 | 0.95 | 0.86 | 0.91 |
| QT | 0.97 | 0.95 | 0.93 | 0.87 | 0.90 |

Here `P.A.` means *Prediction Accuracy*.

Other Requirements
==============

Python packages `nltk` and `argparse` are needed.