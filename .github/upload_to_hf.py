from huggingface_hub import create_repo, upload_file
import os

token = os.getenv("HF_TOKEN")

repo_id = "AvinnaSundar/Ml_model_testing"   # change this

# Create repo if not exists
create_repo(repo_id, token=token, exist_ok=True)

# Upload model.pkl
upload_file(
    path_or_fileobj="iris_model.pkl",
    path_in_repo="iris_model.pkl",
    repo_id=repo_id,
    token=token
)

print("Model uploaded to Hugging Face successfully!")