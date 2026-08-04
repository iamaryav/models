# FFN/ binary classifier using MLP arch
# it's a classic approach

import torch
import torch.nn as nn

vocab_size = 1024
hidden_size = 2048
out = 1

# ----------------------------------------------------------------------------
spam_data = {
  "Meeting tomorrow at 10am": 0,
  "Project deadline update": 0,
  "You won a free iPhone!": 1,
  "Congratulations! You've won $1,000,000": 1,
  "Can you review the PR?": 0,
  "URGENT: Verify your bank account": 1,
  "Lunch plans for Friday": 0,
  "Click here for a free gift card": 1,
  "Team standup notes": 0,
  "Cheap medications online - 90% off": 1,
  "Q3 report attached": 0,
  "You are selected for a cash prize": 1,
  "Happy birthday!": 0,
  "Make money fast - work from home": 1,
  "Sprint retrospective summary": 0
}
# ----------------------------------------------------------------------------

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weights.T + self.bias

class Adam:
    def __init__(self, params, beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=1e-3, eps=1e-8):
        self.params = list(params)
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params): 
            if p.grad is None:
                continue
            grad = p.grad
            # these add_, mul_ and addcmul_ are in place operations
            # p.data.mul_(1 - self.lr * self.weight_decay)
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)

def relu(logits):
    return torch.clamp(logits, min=0)



def sigmoid(logits):
    return 1.0 / (1.0 + torch.exp(-logits))

def loss_calculation(logits, y):
    sig = sigmoid(logits)
    sig = torch.clamp(sig, 1e-7, 1-1e-7)
    loss = -(y * torch.log(sig) + ((1 - y) * torch.log(1 - sig)))
    return loss


class SpamFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Linear(vocab_size, hidden_size)
        self.hidden_layer = Linear(hidden_size, out)

    def forward(self, input, target=None):
        # x = input.shape
        input = self.mlp(input)
        input = relu(input)
        logits = self.hidden_layer(input)
        loss = None
        if target is not None:
            loss = loss_calculation(logits, target)
        return logits, loss

    def predict(self, input):
        # x = input.shape
        logits, _ = self(input, None)
        prob = sigmoid(logits)
        return 1 if prob >= 0.5 else 0

# ----------------------------------------------------------------------------
# bag of word techinque
# build vocab
words = set()
for msg in spam_data:
    words.update(msg.lower().split())
word_to_idx = {w: i for i, w in enumerate(sorted(words))}

# tokenize text
def tokenize(text):
    vec = torch.zeros(vocab_size)
    for word in text.lower().split():
        if word in word_to_idx:
            vec[word_to_idx[word]] = 1
    return vec

# get input function
def get_input(index):
    msg = list(spam_data.keys())[index]
    label = list(spam_data.values())[index]
    return tokenize(msg), torch.tensor(label, dtype=torch.float)
    
# ----------------------------------------------------------------------------

model = SpamFilter()
# adam_optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
adam_optim = Adam(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)

for i in range(1000):
    step = i
    i = i % 15
    x, y = get_input(i)
    adam_optim.zero_grad()
    logits, loss = model(x, y)
    loss = loss.mean()
    loss.backward()
    adam_optim.step()
    print(f"step: {step} | loss: {loss:.4f}")

# ----------------------------------------------------------------------------

is_spam = model.predict(tokenize("you won a bank"))
if is_spam:
    print(f"Spam")
else:
    print(f"Not spam")

# ----------------------------------------------------------------------------