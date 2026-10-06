from datasets import load_dataset

class WildCatDataset:
    def __init__(self):
        self.dataset = load_dataset("allenai/WildChat")
        self.dataset = self.preprocess()

    def preprocess(self):
        print("Loading single turn queries...")
        single_turn = self.dataset['train'].filter(lambda x:x["turn"] == 1)
        print("Filtering only user queries...")
        single_turn_user_queries = single_turn.map(
            lambda x: {
                "conversation_id": x["conversation_id"],
                "content": next(
                    msg["content"]
                    for msg in x["conversation"]
                    if msg["role"] == "user"
                )
            },
            remove_columns=single_turn.column_names
        )
        return single_turn_user_queries
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)