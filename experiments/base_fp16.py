import argparse
import random
import time
import torch
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer

CORPUS = """'''Hugging Face, Inc.''' is an American company that develops tools for building applications using [[machine learning]].<ref>{{Cite web |title=Hugging Face – The AI community building the future. |url=https://huggingface.co/ |access-date=2022-08-20 |website=huggingface.co}}</ref> It is most notable for its Transformers library built for [[natural language processing]] applications and its platform that allows users to share machine learning models and datasets.

== History ==
The company was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf originally as a company that developed a chatbot app targeted at teenagers.<ref>{{Cite web |title=Hugging Face wants to become your artificial BFF |url=https://social.techcrunch.com/2017/03/09/hugging-face-wants-to-become-your-artificial-bff/ |access-date=2022-08-20 |website=TechCrunch |language=en-US}}</ref> After open-sourcing the model behind the chatbot, the company [[Lean startup|pivoted]] to focus on being a platform for democratizing machine learning.

In March 2021, Hugging Face raised $40 million in a [[Series B]] funding round.<ref>{{cite web |title=Hugging Face raises $40 million for its natural language processing library |url=https://techcrunch.com/2021/03/11/hugging-face-raises-40-million-for-its-natural-language-processing-library}}</ref>

On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model.<ref>{{cite web |date=10 January 2022 |title=Inside BigScience, the quest to build a powerful open language model |url=https://venturebeat.com/2022/01/10/inside-bigscience-the-quest-to-build-a-powerful-open-language-model/}}</ref> In 2022, the workshop concluded with the announcement of BLOOM, a multilingual large [[language model]] with 176 billion parameters.<ref>{{Cite web |title=BLOOM |url=https://bigscience.huggingface.co/blog/bloom |access-date=2022-08-20 |website=bigscience.huggingface.co}}</ref>

On December 21, 2021, the company announced its acquisition of Gradio, a software library used to make interactive browser demos of machine learning models.<ref>{{Cite web |title=Gradio is joining Hugging Face! |url=https://huggingface.co/blog/gradio-joins-hf |access-date=2022-08-20 |website=huggingface.co}}</ref>

On May 5, 2022, the company announced its [[Series C]] funding round led by [[Coatue Management|Coatue]] and [[Sequoia fund|Sequoia]].<ref>{{Cite web |last=Cai |first=Kenrick |title=The $2 Billion Emoji: Hugging Face Wants To Be Launchpad For A Machine Learning Revolution |url=https://www.forbes.com/sites/kenrickcai/2022/05/09/the-2-billion-emoji-hugging-face-wants-to-be-launchpad-for-a-machine-learning-revolution/ |access-date=2022-08-20 |website=Forbes |language=en}}</ref> The company received a $2 billion valuation.

On May 13, 2022, the company introduced its Student Ambassador Program to help fulfill its mission to teach machine learning to 5 million people by 2023.<ref>{{Cite web |title=Student Ambassador Program’s call for applications is open! |url=https://huggingface.co/blog/ambassadors |access-date=2022-08-20 |website=huggingface.co}}</ref>

On May 26, 2022, the company announced a partnership with [[Graphcore]] to optimize its Transformers library for the Graphcore IPU.<ref>{{Cite web |title=Graphcore and Hugging Face Launch New Lineup of IPU-Ready Transformers |url=https://huggingface.co/blog/graphcore-update |access-date=2022-08-19 |website=huggingface.co}}</ref>

On August 3, 2022, the company announced the Private Hub, an enterprise version of its public Hugging Face Hub that supports [[Software as a service|SaaS]] or [[On-premises software|on-premise]] deployment.<ref>{{Cite web |title=Introducing the Private Hub: A New Way to Build With Machine Learning |url=https://huggingface.co/blog/introducing-private-hub |access-date=2022-08-20 |website=huggingface.co}}</ref>

== Services and technologies ==
=== Transformers Library ===
The Transformers library is a [[Python (programming language)|Python]] package that contains open-source implementations of [[Transformer (machine learning model)|transformer]] models for text, image, and audio tasks. It is compatible with the [[PyTorch]], [[TensorFlow]] and [[Google JAX|JAX]] [[deep learning]] libraries and includes implementations of notable models like [[BERT (language model)|BERT]] and [[GPT-2|GPT]].<ref>{{Cite web |title=🤗 Transformers |url=https://huggingface.co/docs/transformers/index |access-date=2022-08-20 |website=huggingface.co}}</ref>


=== Hugging Face Hub ===
The Hugging Face Hub is a platform where users can share pretrained datasets, models, and demos of machine learning projects.<ref>{{Cite web |title=Hugging Face Hub documentation |url=https://huggingface.co/docs/hub/index |access-date=2022-08-20 |website=huggingface.co}}</ref> The Hub contains [[GitHub]]-inspired features for code-sharing and collaboration, including discussions and pull requests for projects. It also hosts Hugging Face Spaces, a hosted service that allows users to build web-based demos of machine learning apps using the Gradio or Streamlit.

== References ==
{{Reflist}}

{{Portal bar|Companies}}

{{DEFAULTSORT:Hugging Face}}
[[Category:Machine learning]]
[[Category:Open-source artificial intelligence]]

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a tiny corpus for masked LM")
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=100)

    args = parser.parse_args()
    return args

class DataLoader():
    def __init__(self, tokenizer, batch_size=8, num_batches=100, seq_len=128):
        self.tokenized_corpus = tokenizer(CORPUS).input_ids
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seq_len = seq_len
        self.mask_token_id = tokenizer.mask_token_id
    
    def __iter__(self):
        for _ in range(self.num_batches):
            masked_samples = []
            samples = []
            for _ in range(self.batch_size):
                start = random.randint(0, len(self.tokenized_corpus) - self.seq_len - 1)
                tokens = self.tokenized_corpus[start: start + self.seq_len]
                samples.append(tokens)

                masked_tokens = [(t if random.random() < 0.8 else self.mask_token_id) for t in tokens]
                masked_samples.append(masked_tokens)


            yield {"input_ids": torch.tensor(masked_samples), "labels": torch.tensor(samples)}
    
    def __len__(self):
        return self.num_batches


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    train_dl = DataLoader(tokenizer, batch_size=args.batch_size, num_batches=args.num_batches)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).train()

    start_time = time.time()
    for step, batch in enumerate(train_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 0:
            first_step_time = time.time() - start_time

    total_training_time = time.time() - start_time
    avg_iteration_time = (total_training_time - first_step_time) / (len(train_dl) - 1)
    print("Training finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Average time after the first iteration: {avg_iteration_time * 1000:.2f}ms")

if __name__ == "__main__":
    main()
