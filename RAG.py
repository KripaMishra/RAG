import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class RAGModel:
    def __init__(self, model_name="gpt2", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.embedding_model.to(self.device)
        
        self.index = None
        self.documents = []

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text):
        # Tokenizing input text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
       
        if 'input_ids' not in inputs or 'attention_mask' not in inputs:
            raise ValueError("Tokenized inputs do not contain 'input_ids' or 'attention_mask'")
        
        if inputs['input_ids'].size(0) == 0:
            raise ValueError("Tokenized input_ids is empty")
        if not inputs['input_ids'].dtype == torch.long:
            inputs['input_ids'] = inputs['input_ids'].long()
        
        vocab_size = self.embedding_model.config.vocab_size
        inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, vocab_size - 1)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            try:
                model_output = self.embedding_model(**inputs)
            except Exception as e:
                raise RuntimeError(f"Error during model forward pass: {str(e)}")
            
        if not hasattr(model_output, 'last_hidden_state') or model_output.last_hidden_state.size(1) == 0:
            raise ValueError("Model output does not contain the expected 'last_hidden_state' or it is empty")
        
        embedding = self.mean_pooling(model_output, inputs['attention_mask']).cpu().numpy()
        
        return embedding.flatten()




    def index_data(self, data):
        self.documents = data
        embeddings = [self.get_embedding(doc) for doc in data]
        
        print(f"Shape of first embedding: {embeddings[0].shape}")
        print(f"Number of embeddings: {len(embeddings)}")
        embedding_dim = embeddings[0].shape[0]
        embeddings = [emb for emb in embeddings if emb.shape[0] == embedding_dim]
        
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        embeddings_array = np.array(embeddings)
        print(f"Shape of embeddings array: {embeddings_array.shape}")
        self.index.add(embeddings_array)

    def retrieve_data(self, query, k=1):
        query_embedding = self.get_embedding(query)
        _, I = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.documents[i] for i in I[0]]

    def generate(self, query, max_length=100):
        context = self.retrieve_data(query)[0]
        input_text = f"You are a helpful ai assistant who addressed the queries based on the context provided. Given below are the context and queries that you should answer\nContext: {context}\nQuery: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.split("Answer:")[-1].strip()
