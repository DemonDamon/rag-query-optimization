import csv
import json

def create_dpo_dataset(csv_path, json_path):
    dpo_dataset = []
    with open(csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            conversations = []
            history_str = row.get("history", "[]")
            original_question = row.get("question", "")
            rewritten_question = row.get("rewrite", "")

            try:
                history_list = json.loads(history_str)
            except (json.JSONDecodeError, TypeError):
                history_list = []

            # Add history turns to the conversations list
            for turn in history_list:
                # As per user feedback, keys could be 'human'/'user'/'q' and 'gpt'/'assistant'/'assitant'/'a'
                user_query = turn.get("human") or turn.get("user") or turn.get("q")
                assistant_answer = turn.get("gpt") or turn.get("assistant") or turn.get("assitant") or turn.get("a")

                if user_query:
                    conversations.append({"from": "human", "value": user_query})
                if assistant_answer:
                    conversations.append({"from": "gpt", "value": assistant_answer})

            # Add the current user query as the last turn
            if original_question:
                conversations.append({"from": "human", "value": original_question})

            # Create the DPO entry
            dpo_entry = {
                "conversations": conversations,
                "chosen": {
                    "from": "gpt",
                    "value": rewritten_question
                },
                "rejected": {
                    "from": "gpt",
                    "value": original_question
                }
            }
            dpo_dataset.append(dpo_entry)

    with open(json_path, mode='w', encoding='utf-8') as outfile:
        json.dump(dpo_dataset, outfile, ensure_ascii=False, indent=2)
    print(f"DPO dataset created and saved to {json_path}")

if __name__ == "__main__":
    csv_path = "/Users/damon/myWork/rag-query-optimization/data/query_rewrite_dataset.csv"
    json_path = "/Users/damon/myWork/rag-query-optimization/data/dpo_query_rewrite_dataset.json"
    create_dpo_dataset(csv_path, json_path)