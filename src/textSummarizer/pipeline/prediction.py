from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        #from documentation:
        #summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        pipe = pipeline("summarization", model=self.config.model_path,tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, length_penalty = 0.8, num_beams = 8, max_length = 128)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output
