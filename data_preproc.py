import csv
import json

def process_data(cur_split):
    # load raw text
    with open(f"./data/raw/QuALITY.v1.0.1.htmlstripped.{cur_split}", "r") as f:
        data = f.readlines()
    data = [json.loads(x) for x in data]

    # extract questions
    ret = []
    cnter = 0
    for item in data:
        article_id = item['article_id']
        for q in item['questions']:
            question = q['question']
            options = q['options']
            gold_label = q['gold_label']
            assert len(options) == 4
            q_item = {
                'qid': cnter,
                'article_id' : article_id,
                'question' : question,
                'option_1' : options[0],
                'option_2' : options[1],
                'option_3' : options[2],
                'option_4' : options[3],
                'gold_label' : gold_label,
            }
            ret.append(q_item)
            cnter += 1

    with open(f"./data/processed/quality_{cur_split}_q.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'article_id', 'question', 'option_1',
        'option_2', 'option_3', 'option_4', 'gold_label'])
        writer.writeheader()
        writer.writerows(ret)

if __name__ == "__main__":
    process_data("train")
    process_data("dev")