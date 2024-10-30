import logging
from transformers import pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

class QuestionAnsweringAgent:
    def __init__(self):
        self.qa_model = pipeline("question-answering",
                                 model="distilbert-base-uncased-distilled-squad",
                                 top_k=1)

    def query(self, query, context_df):
        if context_df.empty:
            return "No context data available."

        top_columns = context_df.columns

        context = ""
        for _, row in context_df.iterrows():
            node_label = row.get("node label", "")

            row_context = f"This text is about \"{node_label}\":\n"

            for col in top_columns:
                if col == "node label":
                    continue

                values = row[col]
                values_lst = str(values).split(",")

                if len(values_lst) > 5:
                    row_context += f"{col}: {', '.join(values_lst[:5])}\n"
                else:
                    row_context += f"{col}: {', '.join(values_lst)}\n"

            context += row_context + "\n"

        output = self.qa_model(question=query, context=context)

        if isinstance(output, list) and output:
            answer_str = ", ".join([result['answer'] for result in output])
        elif isinstance(output, dict):
            answer_str = output.get('answer', '')
        else:
            answer_str = "No answer found."

        if not answer_str:
            answer_str = "No answer found."

        return answer_str
