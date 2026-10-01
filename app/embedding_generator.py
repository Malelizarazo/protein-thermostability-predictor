import torch
from transformers import EsmTokenizer, EsmModel

class EmbeddingGenerator:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", token=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name, token=token)
        self.model = EsmModel.from_pretrained(model_name, token=token).to(self.device)
        self.model.eval()

    def generate_embeddings(self, sequences, protein_ids, batch_size=1):
        embeddings = []
        valid_protein_ids = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            for j in range(len(batch)):
                with torch.no_grad():
                    output = self.model(**{k: v[j:j+1] for k, v in inputs.items()})
                    embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding.squeeze(0))
                    valid_protein_ids.append(protein_ids[j])

        return embeddings, valid_protein_ids
