
import re
import json
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import logging
import argparse

from tie_recommend import TIEModel
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews_file", required=True, help="the input JSON file containing the reviews")
    parser.add_argument("--output_file", required=True, help="the file to output the results")
    parser.add_argument("--model_file", required=True, help="the file to store the model")
    parser.add_argument("--max_reviews", required=False, help="the maximum number of reviews to be processed", type=int)
    parser.add_argument("--alpha", required=False, help="parameter `alpha` in TIE paper, which defaults to 0.7", type=float)
    args = parser.parse_args()

    json_file = open(args.reviews_file, 'r')
    reviews = json.loads(json_file.read())
    json_file.close()

    stemmer = LancasterStemmer()

    # auxiliary functions

    def get_all_reviewers(reviews):
        reviewer_set = set()
        for review in reviews:
            for reviewer in review["reviewers"]:
                reviewer_set.add(reviewer["id"])
        return list(reviewer_set)

    def is_word_useful(word):
        for c in word:
            if c.isdigit():
                return False
        if "http://" in word or "https://" in word:
            return False
        return True

    def word_stem(word):
        if word.endswith('.') or word.endswith(',') or word.endswith(':') or word.endswith('\'') or word.endswith('\"'):
            word = word[:-1]
        if word.startswith(',') or word.startswith('.') or word.startswith(':') or word.startswith('\'') or word.startswith('\"'):
            word = word[1:]
        return stemmer.stem(word)

    def split_text(txt):
        splitted_words = list(
            map(lambda x: word_stem(x),
                filter(lambda x: is_word_useful(x), re.split(r"[\s\n\t]+", txt))
            )
        )
        return splitted_words

    def get_all_words(reviews):
        s = set()
        for review in reviews:
            for w in split_text(review["textual-content"]):
                s.add(w)
        l = list(s)
        return l

    alpha = 0.7 if args.alpha is None else args.alpha
    
    model = TIEModel(word_list=get_all_words(reviews), reviewer_list=get_all_reviewers(reviews), alpha=alpha, M=50, text_splitter=split_text)

    mrr_accumulation = 0
    is_recomm_accumulation_top_10 = 0
    is_recomm_accumulation_top_5 = 0
    is_recomm_accumulation_top_3 = 0
    is_recomm_accumulation_top_1 = 0
    current_predicted = 0

    i = 0
    result_obj = {
        "recommendation-results": []
    }
    max_reviews = args.max_reviews if args.max_reviews is not None else len(reviews)
    while i < min(len(reviews), max_reviews):
        if i + 1 == len(reviews):
            break
        model.update(reviews[i])
        recommended_reviewers = model.recommend(reviews[i + 1], max_count=1000)
        result_obj["recommendation-results"].append({"review-id": reviews[i + 1]["id"],
        "result": recommended_reviewers[:10]})
        actual_reviewers = list(map(lambda x: x["id"], reviews[i + 1]["reviewers"]))
        
        logging.info("Progress: {}/{} reviews".format(i + 1, len(reviews)))
        logging.info("ID: {}".format(reviews[i + 1]["id"]))
        logging.info("Recommended: {}".format(recommended_reviewers[:10]))
        logging.info("Actual: {}".format(actual_reviewers))
        current_predicted += 1

        is_recomm_top_10 = 0
        is_recomm_top_5 = 0
        is_recomm_top_3 = 0
        is_recomm_top_1 = 0
        for r in actual_reviewers:
            if r in recommended_reviewers[:10]:
                is_recomm_top_10 = 1
            if r in recommended_reviewers[:5]:
                is_recomm_top_5 = 1
            if r in recommended_reviewers[:3]:
                is_recomm_top_3 = 1
            if r == recommended_reviewers[0]:
                is_recomm_top_1 = 1
        is_recomm_accumulation_top_10 += is_recomm_top_10
        is_recomm_accumulation_top_5 += is_recomm_top_5
        is_recomm_accumulation_top_3 += is_recomm_top_3
        is_recomm_accumulation_top_1 += is_recomm_top_1

        rank = int(1e8)
        for k in range(len(recommended_reviewers)):
            if recommended_reviewers[k] in actual_reviewers:
                rank = k
                break
        mrr_accumulation += 1 / (rank + 1)

        top10acc = is_recomm_accumulation_top_10 / current_predicted
        top5acc = is_recomm_accumulation_top_5 / current_predicted
        top3acc = is_recomm_accumulation_top_3 / current_predicted
        top1acc = is_recomm_accumulation_top_1 / current_predicted
        mrr_val = mrr_accumulation / current_predicted

        result_obj["top10-accuracy"] = round(top10acc, 2)
        result_obj["top5-accuracy"] = round(top5acc, 2)
        result_obj["top3-accuracy"] = round(top3acc, 2)
        result_obj["top1-accuracy"] = round(top1acc, 2)
        result_obj["mrr"] = round(mrr_val, 2)

        logging.info('Top-10 Predict Accuracy: %.6f', top10acc)
        logging.info('Top-5 Predict Accuracy: %.6f', top5acc)
        logging.info('Top-3 Predict Accuracy: %.6f', top3acc)
        logging.info('Top-1 Predict Accuracy: %.6f', top1acc)
        logging.info('MRR: %.6f', mrr_val)
        model.update(reviews[i + 1])
        i += 2
    
    output_file = open(args.output_file, 'w')
    output_file.write(json.dumps(result_obj))
    output_file.close()
    logging.info('Recommendation results have been written to {}.'.format(args.output_file))
    model.save(args.model_file)
    logging.info('Model file has been written to {}.'.format(args.model_file))