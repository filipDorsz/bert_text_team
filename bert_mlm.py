import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    BertTokenizer, 
    BertForMaskedLM, 
    DataCollatorForLanguageModeling, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy

# Import the dataset class (Ensure dataset.py exists in the same directory)
from text_unlabeled_dataset import UnlabeledTextDataset

def save_plot(train_loss, val_loss, output_dir):
    """Saves a high-quality loss graph comparing Train vs Validation."""
    try:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
        plt.figure(figsize=(10, 6))
        
        # Plot Training Loss
        # We assume train_loss is recorded more frequently, so we smooth it or just plot it
        # For clarity, let's plot distinct points if the lists are short (epochs)
        epochs = np.arange(1, len(train_loss) + 1)
        
        sns.lineplot(x=epochs, y=train_loss, linewidth=2.5, label='Training Loss', marker='o')
        sns.lineplot(x=epochs, y=val_loss, linewidth=2.5, label='Validation Loss', marker='o', color='orange')
        
        plt.title('BERT Domain Adaptation (MLM)', fontsize=18, weight='bold', pad=15)
        plt.xlabel('Epochs', fontsize=14, labelpad=10)
        plt.ylabel('Cross-Entropy Loss', fontsize=14, labelpad=10)
        plt.legend()
        
        sns.despine()
        
        plot_path = os.path.join(output_dir, "bert_pretrain_loss_paper.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {plot_path}")
    except Exception as e:
        print(f"Error plotting: {e}")

def visualize_predictions(model, tokenizer, dataset, device, num_samples=5):
    """
    Randomly selects samples from the dataset, masks a word, and shows top 10 predictions.
    """
    print(f"\n{'='*20} QUALITATIVE EVALUATION {'='*20}")
    print(f"Showing Top-10 predictions for {num_samples} random samples from Validation Set...")
    
    model.eval()
    
    # Handle Subset (created by random_split) or regular Dataset
    dataset_len = len(dataset)
    indices = np.random.choice(dataset_len, size=min(num_samples, dataset_len), replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample
        sample = dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0).to(device) # Shape [1, seq_len]
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        
        # Determine sequence length (ignoring padding)
        active_len = attention_mask.sum().item()
        
        # Pick a random position to mask (skipping [CLS] at 0 and [SEP] at active_len-1)
        # We need at least 3 tokens ([CLS], word, [SEP]) to mask something
        if active_len < 3:
            continue
            
        mask_pos = np.random.randint(1, active_len - 1)
        
        # Store original
        original_token_id = input_ids[0, mask_pos].item()
        original_word = tokenizer.decode([original_token_id])
        
        # Mask the token
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, mask_pos] = tokenizer.mask_token_id
        
        # Predict
        with torch.no_grad():
            outputs = model(masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits # [1, seq_len, vocab_size]
            
        # Get probabilities for the masked position
        mask_logits = logits[0, mask_pos, :]
        probs = torch.softmax(mask_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, 10)
        
        # Display
        print(f"\n--- Sample {i+1} ---")
        
        # Decode surrounding context for readability
        # Get a window around the mask
        start = max(0, mask_pos - 10)
        end = min(active_len, mask_pos + 11)
        context_ids = input_ids[0, start:end].tolist()
        
        # Replace the target with [MASK] for display
        display_ids = context_ids.copy()
        display_ids[mask_pos - start] = tokenizer.mask_token_id
        context_text = tokenizer.decode(display_ids)
        
        print(f"Context: \"...{context_text}...\"")
        print(f"Masked Word: '{original_word}'")
        print(f"Predictions:")
        
        for prob, idx in zip(top_probs, top_indices):
            token_str = tokenizer.decode([idx])
            marker = "  <-- CORRECT" if idx.item() == original_token_id else ""
            print(f"  {token_str:<15} ({prob.item():.2%}) {marker}")
            
    print(f"{'='*60}\n")

def train_model():
    # --- Configuration ---
    EPOCHS = 10                  # Set high, Early Stopping will cut it short
    PATIENCE = 2                 # Stop if val loss doesn't improve for 3 epochs
    BATCH_SIZE = 32              # Low batch size per GPU
    GRAD_ACCUM_STEPS = 1         # Effective batch size = 32
    LEARNING_RATE = 5e-5
    WARMUP_RATIO = 0          
    MAX_LEN = 512
    MODEL_NAME = 'bert-base-uncased'
    
    # Adjust paths as needed
    JSON_PATH = os.path.join('data', 'text_data_all.json')
    OUTPUT_DIR = './finetuned_bert_mlm'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 1. Setup Tokenizer & Dataset ---
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    print("Loading dataset...")
    full_dataset = UnlabeledTextDataset(
        json_path=JSON_PATH, 
        tokenizer=tokenizer, 
        max_length=MAX_LEN
    )

    # --- 2. Create Validation Split (CRITICAL) ---
    # Split 90% Train / 10% Validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset Split: {train_size} Training | {val_size} Validation")

    # --- 3. Setup Data Collator ---
    # Note: Standard MLM is used here for compatibility with pre-tokenized 'input_ids'.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    def custom_collate_fn(batch):
        # Filter out any None items if dataset has errors
        batch = [item for item in batch if item is not None]
        clean_batch = []
        for sample in batch:
            clean_batch.append({
                'input_ids': sample['input_ids'],
                'attention_mask': sample['attention_mask']
            })
        return data_collator(clean_batch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=2
    )
    
    # Validation dataloader (No shuffle, bigger batch size is okay since no gradients)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=custom_collate_fn, num_workers=2
    )

    # --- 4. Setup Model, Optimizer, Scheduler, Scaler ---
    print("Loading BERT for Masked LM...")
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Linear Warmup -> Linear Decay
    total_steps = (len(train_dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    # Mixed Precision Scaler
    scaler = GradScaler()

    # --- 5. Training Loop with Early Stopping ---
    print(f"Starting training for {EPOCHS} epochs with Early Stopping...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        running_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1} [Train]")

        for step, batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Autocast for Mixed Precision (Speed + Memory)
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            
            # Record RAW loss (fix math bug)
            running_loss += outputs.loss.item()
            
            # Step Optimizer
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                current_steps = (step + 1) % GRAD_ACCUM_STEPS if (step + 1) % GRAD_ACCUM_STEPS != 0 else GRAD_ACCUM_STEPS
                avg_step_loss = running_loss / current_steps
                
                total_train_loss += running_loss # Accumulate unscaled loss
                progress_bar.set_postfix({'loss': f'{avg_step_loss:.4f}'})
                running_loss = 0

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_train_losses.append(avg_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        print(f"Running Validation...")
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        epoch_val_losses.append(avg_val_loss)
        
        print(f"\nResults Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        # --- EARLY STOPPING CHECK ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New Best Model! Saving checkpoint...")
            
            # Save strictly the best model
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
            break

    # --- 6. Wrap Up ---
    # Reload best weights
    if best_model_state is not None:
        print("Restoring best model weights...")
        model.load_state_dict(best_model_state)
        
    print("Generating report...")
    save_plot(epoch_train_losses, epoch_val_losses, OUTPUT_DIR)
    
    # NEW: Run Qualitative Evaluation
    visualize_predictions(model, tokenizer, val_dataset, device)
    
    print("Training pipeline complete.")

if __name__ == "__main__":
    train_model()