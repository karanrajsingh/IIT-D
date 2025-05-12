import os
import json
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet101
from torchvision import transforms
from transformers import BertTokenizer
from transformers import get_cosine_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim=768, num_layers=6, nhead=8):
        super(TextEncoder, self).__init__()        
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, max_len + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, input_ids, attention_mask= None):
        batch_size, seq_len = input_ids.size()        
        token_embeds = self.token_embeddings(input_ids)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, token_embeds], dim=1)
        self.dropout = nn.Dropout(0.1)
        embeddings = self.dropout(embeddings + self.position_embeddings[:, :seq_len + 1, :])
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(batch_size, 1).to(input_ids.device), attention_mask], dim=1)
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None        
        encoded_output = self.transformer_encoder(embeddings, src_key_padding_mask= src_key_padding_mask)
        cls_output = encoded_output[:, 0, :]        
        return cls_output

class ImageEncoder(nn.Module):
    def __init__(self, dvembed=768):
        super(ImageEncoder, self).__init__()
        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.proj = nn.Linear(2048, dvembed)
        
    def forward(self, images):
        features = self.resnet(images)
        B, C, h, w = features.size()
        features = features.permute(0, 2, 3, 1).reshape(B, h*w, C)
        features = self.proj(features)
        return features

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, text_features, image_features):
        attn_output, attn_weights = self.cross_attention(text_features, image_features, image_features)
        fused_features = attn_output[:,0:1,:]
        self.dropout = nn.Dropout(0.1)
        fused_features = self.dropout(fused_features.squeeze(1))        
        return fused_features
    
class Classifier(nn.Module):
    def __init__(self, d_embed=768, hidden_dim=500, num_classes=28):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(d_embed, hidden_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes))
        
    def forward(self, x):
        return self.mlp(x)

class VQAModel(nn.Module):
    def __init__(self, vocab_size, max_len, num_classes):
        super(VQAModel, self).__init__()       
        self.text_encoder = TextEncoder(vocab_size, max_len)
        self.image_encoder = ImageEncoder()
        self.cross_attention = CrossAttentionFusion()
        self.classifier = Classifier(num_classes=num_classes)
        
    def forward(self, input_ids, images, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)        
        text_features = text_features.unsqueeze(1)
        fused_features = self.cross_attention(text_features, image_features)       
        logits = self.classifier(fused_features)
        return logits

def train_model(model, train_loader, val_loader, optimizer, scheduler, start_epoch, num_epochs=10, best_val_acc = 0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  
    best_val_loss = float('inf')
    acc_loss_df = []  
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0       
        for images, input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", ncols=100):
            input_ids, images, labels, attention_mask = input_ids.to(device), images.to(device), labels.to(device), attention_mask.to(device)           
            optimizer.zero_grad()
            outputs = model(input_ids, images, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()*labels.size(0)
            t, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)        
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)        
        acc_loss = {"Epoch" :epoch+1, "Train Loss": train_loss, "Train Acc" : train_acc, "Val Loss" : val_loss, "Val Acc" : val_acc}
        acc_loss_df.append(acc_loss)
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"Epoch {epoch+1} | LR: {current_lr:.8f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        save_checkpoint = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint = True        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint = True        
        if save_checkpoint:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss' : best_val_loss
            }, f"checkpoint_epoch{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")           
    df = pd.DataFrame(acc_loss_df)
    df.to_csv(base_path/f'Loss_Accuracy_parameters.csv')
def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0    
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in data_loader:
            input_ids, images, labels, attention_mask = input_ids.to(device), images.to(device), labels.to(device), attention_mask.to(device)
            outputs = model(input_ids, images, attention_mask)
            loss = criterion(outputs, labels)            
            running_loss += loss.item()*labels.size(0)
            t, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)    
    return running_loss / total, correct / total

class CLEVRVQADataset(Dataset):
    def __init__(self, image_dir, question_path, tokenizer, transform=None, max_len = 48):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(question_path, 'r') as f:
            questions_json = json.load(f)        
        self.questions = questions_json['questions']
        all_answers = list({q['answer'] for q in self.questions})
        self.answer_to_idx = {ans: idx for idx, ans in enumerate(sorted(all_answers))}
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_to_idx.items()}

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        image_filename = q['image_filename']
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        question = q['question']
        encoding = self.tokenizer(question, padding='max_length', max_length=self.max_len, truncation= True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        answer = q['answer']
        answer_idx = self.answer_to_idx[answer]
        return (image, input_ids, attention_mask, torch.tensor(answer_idx))

base_path = Path().resolve()
img_dir = base_path/'CLEVR_COL774_A4'/'images'
train_img_dir = img_dir/'trainA'
val_img_dir = img_dir/'valA'
test_img_dir = img_dir/'testA'
q_json = base_path/'CLEVR_COL774_A4'/'questions'
train_q_json = q_json/'CLEVR_trainA_questions.json'
val_q_json = q_json/'CLEVR_valA_questions.json'
test_q_json = q_json/'CLEVR_testA_questions.json'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
image_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
num_classes = 28
batch_size = 512
vocab_size = tokenizer.vocab_size
max_len = 48
train_dataset = CLEVRVQADataset(image_dir=train_img_dir, question_path=train_q_json, tokenizer=tokenizer, transform=image_transform, max_len= max_len)
val_dataset = CLEVRVQADataset(image_dir=val_img_dir, question_path=val_q_json, tokenizer=tokenizer, transform=image_transform, max_len= max_len )
test_dataset = CLEVRVQADataset(image_dir=test_img_dir, question_path=test_q_json, tokenizer=tokenizer, transform=image_transform, max_len= max_len)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=32)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=32)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=32)

model = VQAModel(vocab_size=vocab_size, max_len=max_len, num_classes=num_classes)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
best_val_loss = float('inf')
best_val_acc = 0.0
num_epochs = 17
start_epoch = 0
checkpoint_path = base_path/'checkpoint_epoch11.pth'
total_steps = num_epochs*len(train_loader)
warmup_steps = int(0.1*total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    print(f"Resuming training from epoch {start_epoch}")
train_model(model= model, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, scheduler = scheduler, start_epoch = start_epoch, num_epochs=num_epochs, best_val_acc = best_val_acc)
