Introdução
Olá, Yuan! Entendi que você quer treinar um modelo "do zero" (from scratch), mas em uma versão muito muito simples da arquitetura do Qwen3. Isso faz sentido para fins educacionais, pois o modelo original de 0.6B é grande e requer muitos dados/hardware para treinar de verdade. Vou simplificar drasticamente:

Manter a mesma estrutura: Transformer decoder com RoPE, Grouped Query Attention (GQA), SiLU no FFN, RMSNorm, embeddings compartilhados.
Simplificações extremas:
Dimensão de embedding (dim): 64 (em vez de 1024).
Cabeças de atenção: 4 para Query (2 para KV no GQA).
Camadas (layers): 2 (em vez de 28).
Dimensão oculta no FFN: 128 (em vez de 3072).
Vocabulário: Muito pequeno (~100 tokens, baseado em caracteres para simplicidade).
Contexto máximo: 128 tokens (em vez de 4k).
Parâmetros totais: ~10k (muito menor que 0.6B, para rodar rápido em CPU).


Isso resulta em um "mini-Qwen" que pode ser treinado em minutos com dados simples. Usaremos PyTorch puro. Para treinamento: um texto hardcoded simples (ex: uma frase repetida), tokenizer básico por caracteres, e um loop de treinamento com Adam e CrossEntropyLoss.
Aviso: Isso é só para aprendizado. O modelo aprenderá padrões básicos, não linguagem real. Para datasets reais, use Hugging Face Datasets. Rode em CPU ou GPU. Instruções em português.
Passo 1: Importar bibliotecas
Instale PyTorch se necessário: pip install torch. Código:
Pythonimport torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
Passo 2: Implementar RMSNorm (mesmo do original, mas simples)
Pythonclass RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        norm = x / torch.sqrt(var + self.eps)
        return norm * self.weight
Passo 3: Implementar RoPE (simplificado)
Função para frequências:
Pythondef precompute_rope_freqs(dim: int, max_seq_len: int = 128, base: float = 10000.0) -> tuple:
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    m = torch.arange(max_seq_len).unsqueeze(1)
    freqs = m * theta.unsqueeze(0)
    return freqs.cos(), freqs.sin()
Aplicar RoPE:
Pythondef apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)
Passo 4: Implementar Grouped Query Attention (GQA simplificado)
Pythonclass GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, max_seq_len: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.group_size = num_heads // num_kv_heads
        
        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.cos, self.sin = precompute_rope_freqs(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        pos = torch.arange(seq_len, device=x.device)
        cos = self.cos[pos].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        sin = self.sin[pos].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        k = k.unsqueeze(1).repeat(1, self.group_size, 1, 1, 1).view(batch, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(1).repeat(1, self.group_size, 1, 1, 1).view(batch, self.num_heads, seq_len, self.head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)
Passo 5: Implementar Feed Forward com SiLU
Pythonclass FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        val = self.w2(x)
        return self.w3(gate * val)
Passo 6: Implementar Bloco Transformer
Pythonclass TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, hidden_dim: int, max_seq_len: int = 128):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
Passo 7: Implementar o Modelo Mini-Qwen
Pythonclass MiniQwen(nn.Module):
    def __init__(self, vocab_size: int = 100, dim: int = 64, num_layers: int = 2,
                 num_heads: int = 4, num_kv_heads: int = 2, hidden_dim: int = 128,
                 max_seq_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, num_kv_heads, hidden_dim, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Tie embeddings

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(input_ids)
        if mask is None:
            mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1], device=input_ids.device) * float('-inf'), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return self.lm_head(x)
Passo 8: Preparar Dados e Tokenizer Simples
Usaremos um tokenizer por caracteres (simples). Texto de exemplo: uma frase repetida para aprender padrões.
Python# Tokenizer simples: mapeia chars para IDs
def build_vocab(text: str) -> dict:
    chars = sorted(list(set(text)))
    return {ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)}

# Dados de exemplo
text = "Olá mundo! Este é um teste simples. Olá mundo! " * 10  # Repetido para mais dados
char_to_id, id_to_char = build_vocab(text + " ")  # Inclui espaço
vocab_size = len(char_to_id)

# Converte texto para tensor
data = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)

# Função para batches
def get_batch(seq_len: int = 32, batch_size: int = 4):
    starts = torch.randint(0, len(data) - seq_len, (batch_size,))
    inputs = torch.stack([data[i:i+seq_len] for i in starts])
    targets = torch.stack([data[i+1:i+seq_len+1] for i in starts])
    return inputs, targets
Passo 9: Treinar o Modelo
Código para treinamento:
Python# Instância do modelo
model = MiniQwen(vocab_size=vocab_size)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Loop de treinamento
epochs = 100  # Pequeno para teste
for epoch in range(epochs):
    inputs, targets = get_batch()
    logits = model(inputs)  # [batch, seq, vocab]
    logits = logits.view(-1, vocab_size)  # Flatten para loss
    targets = targets.view(-1)
    loss = loss_fn(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print("Treinamento concluído!")
Passo 10: Inferência (Gerar Texto)
Após treinar, teste gerando texto:
Pythondef generate(model, seed_text: str, max_tokens: int = 20):
    model.eval()
    input_ids = torch.tensor([char_to_id[c] for c in seed_text], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
    generated = ''.join([id_to_char[id.item()] for id in input_ids[0]])
    return generated

# Exemplo
print(generate(model, "Olá "))
Instruções Finais

Copie todo o código em um arquivo .py e rode python arquivo.py.
O modelo aprenderá a repetir padrões do texto (ex: completar "Olá" com "mundo!").
Tempo: Em CPU, <1 min para 100 epochs.
Para melhorar: Aumente epochs, use texto maior (ex: baixe um livro txt), ajuste hiperparâmetros.
Memória: Muito baixa (~10MB).
Se quiser mais realista, integre com datasets como torchtext.datasets.WikiText2.

Se precisar de ajustes ou mais complexidade, diga! 😊