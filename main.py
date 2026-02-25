import json
import os
import torch
import numpy as np
import pandas as pd
from src.evaluation import evaluate_classification
from src.models import BERTClassifier
from src.training import get_bert_dataloaders
from src.training import train_bert_classifier
from src.constants import MODEL_NAME, classes_dict
np.random.seed(42)
torch.manual_seed(42)


reversed_classes_dict = {v: k for k, v in classes_dict.items()}
"""
use: nohup uv run main.py > output.log 2>&1 &
"""
if __name__ == "__main__":
    #json_path = os.path.join("data", "labeled_text_data.json")
    json_path = "data/labeled_text_data.json" #here we need to have labeled data for training
    # BERT-specific parameters
    batch_size = 16  
    epochs =10
    learning_rate = 5e-5
    num_classes = 3
    layers_unfrozen = 6
    train_mode = True 
    
    custom_bert_path = "./finetuned_bert_mlm" # our path to ssl pretrained model
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Używane urządzenie: {device}")

    dataloader_train, dataloader_test = get_bert_dataloaders(
        json_path, 
        batch_size=batch_size,
        model_path=custom_bert_path
    )

    model_name = f"{MODEL_NAME}_{epochs}_epochs.pth"
    
    if train_mode:
        model = BERTClassifier(num_classes=num_classes, dropout_prob=0.25, pretrain_path=custom_bert_path)

        trained_model = train_bert_classifier(
            model, 
            dataloader_train, 
            dataloader_test,
            epochs=epochs, 
            learning_rate=learning_rate,
            freeze_until_layer=layers_unfrozen
        )
        
        trained_model_cpu = trained_model.to(torch.device("cpu"))
        os.makedirs("models", exist_ok=True)
        torch.save(
            trained_model_cpu.state_dict(),
            os.path.join("models", model_name),
        )
        trained_model = trained_model.to(device)
    else:
        
        trained_model = BERTClassifier(num_classes=num_classes, dropout_prob=0.3)
        state = torch.load(
            os.path.join("models", model_name), map_location=device
        )
        trained_model.load_state_dict(state)
        trained_model = trained_model.to(device)

    trained_model.eval()
    metrics = evaluate_classification(dataloader_test, trained_model)

    print("\nMetryki końcowe (BERT):")
    for metric, value in metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}: \n{value}")
        else:
            print(f"{metric}: {value:.4f}")