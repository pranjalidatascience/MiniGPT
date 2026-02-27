## Building and training a bigram language model
from functools import partial
import math
from multiprocessing import context

import config
import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Class definition for a simple bigram language model.
    """

    def __init__(self, config):
        """
        Initialize the bigram language model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.embeddings)
        2. A linear layer that maps embeddings to logits. (self.linear) **set bias to True**
        3. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        super().__init__()
        # ========= TODO : START ========= #
        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        # ========= TODO : END ========= #

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the bigram language model.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, 1) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, vocab_size) containing the logits.
        """

        # ========= TODO : START ========= #
        # B, T = x.shape
        # tok_emb = self.embeddings(x)  # Shape (B, T, embed_dim)
        # pos_emb=self.positional_embedding_table(
        #     torch.arange(T, device=x.device))  # Shape (T, embed_dim)
        # x = tok_emb + pos_emb  # Add positional embeddings
        
        x=self.embeddings(x)  # Shape (B, T, embed_dim)           
        x=self.dropout(x)
        logits=self.linear(x)
        
        return logits
        
        # raise NotImplementedError

        # ========= TODO : END ========= #

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.
        We will perform multinomial sampling which is very similar to greedy sampling
        but instead of taking the token with the highest probability, we sample the next token from a multinomial distribution.


        Args:
        context : List[int]
            A list of integers (tokens) representing the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Output:
        List[int]
            A list of integers (tokens) representing the generated tokens.
        """

        ### ========= TODO : START ========= ###
        if torch.is_tensor(context):
            generated = context.flatten().tolist()
        else:
            generated = list(context)
            
        for _ in range(max_new_tokens):
            # Pass the sequence to the model (Bigram only cares about the last token)
            # Input shape needs to be (Batch, Seq_len)
            input_tensor = torch.tensor([generated], device=next(self.parameters()).device)
            logits = self.forward(input_tensor)
            
            # Get the probabilities for the last token
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
        # Return as a tensor so .squeeze().tolist() works in the notebook
        return torch.tensor(generated)
        # generated =list(context)
        # for _ in range(max_new_tokens):
        #     # context_cond=context[:,-1:].unsqueeze(0).to(next(self.parameters()).device)  # Shape (1, seq_len)
        #     last_token = generated[-1]
        #     logits=self.forward(torch.tensor([generated]))
        #     probs=F.softmax(logits[0,-1,:],dim=-1)
        #     next_token=torch.multinomial(probs,num_samples=1).item()
        #     generated.append(next_token)
        # return generated
        #     logits, _ =self(context)
        #     logits=logits[:,-1,:]
        #     probs=torch.functional.softmax(logits,dim=-1)
        #     next_token=torch.multinomial(probs,num_samples=1)
        #     context=torch.cat([context,next_token],dim=1)
        # return context[:,1:].tolist()[0]
            # context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)  # Shape (1, seq_len)
            # generated_tokens = []
            # for _ in range(max_new_tokens):
            #     logits = self.forward(context[:, -1:])  # Get logits for the last token
            #     probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
            #     next_token = torch.multinomial(probs, num_samples=1)  # Sample the next token
            #     generated_tokens.append(next_token.item())
            #     context = torch.cat([context, next_token], dim=1)  # Append the new token to the context
            # return generated_tokens

        ### ========= TODO : END ========= ###


class SingleHeadAttention(nn.Module):
    """
    Class definition for Single Head Causal Self Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)

    """

    def __init__(
        self,
        input_dim,
        output_key_query_dim=None,
        output_value_dim=None,
        dropout=0.1,
        max_len=512,
    ):
        """
        Initialize the Single Head Attention Layer.

        The model should have the following layers:
        1. A linear layer for key. (self.key) **set bias to False**
        2. A linear layer for query. (self.query) **set bias to False**
        3. A linear layer for value. (self.value) # **set bias to False**
        4. A dropout layer. (self.dropout)
        5. A causal mask. (self.causal_mask) This should be registered as a buffer.
        NOTE : Please make sure that the causal mask is upper triangular and not lower triangular (this helps in setting up the test cases, )

         NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        if output_key_query_dim:
            self.output_key_query_dim = output_key_query_dim
        else:
            self.output_key_query_dim = input_dim

        if output_value_dim:
            self.output_value_dim = output_value_dim
        else:
            self.output_value_dim = input_dim

        causal_mask = None  # You have to implement this, currently just a placeholder

        # ========= TODO : START ========= #
        self.key = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.query = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.value = nn.Linear(input_dim, self.output_value_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Use torch.triu (upper triangular) for the causal mask as requested in docstrings.
        # diagonal=1 ensures we don't mask the current token itself.
        causal_mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        # ========= TODO : END ========= #

        self.register_buffer(
            "causal_mask", causal_mask
        )  # Registering as buffer to avoid backpropagation

    def forward(self, x):
        """
        Forward pass of the Single Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, output_value_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #
        batch_size, num_tokens, _ = x.shape

        k = self.key(x) 
        q = self.query(x)
        v = self.value(x)

        # Calculate attention scores with proper scaling
        # Use math.sqrt for precision consistent with standard transformer implementations
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])

        # Apply causal mask: mask positions where the upper triangular mask is True (1)
        # These are the "future" tokens that should be invisible to the current token.
        attention_scores = attention_scores.masked_fill(self.causal_mask[:num_tokens, :num_tokens], float('-inf'))

        # Standard softmax, dropout, and value multiplication
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)

        return output
        # ========= TODO : END ========= #


class MultiHeadAttention(nn.Module):
    """
    Class definition for Multi Head Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(self, input_dim, num_heads, dropout=0.1) -> None:
        """
        Initialize the Multi Head Attention Layer.

        The model should have the following layers:
        1. Multiple SingleHeadAttention layers. (self.head_{i}) Use setattr to dynamically set the layers.
        2. A linear layer for output. (self.out) **set bias to True**
        3. A dropout layer. (self.dropout) Apply dropout to the output of the out layer.

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        # ========= TODO : START ========= #
        # Each head handles a fraction of the total embedding dimension
        head_dim = input_dim // num_heads

        # Dynamically create the heads (head_0, head_1, etc.)
        for i in range(num_heads):
            # Crucial: Specify output_key_query_dim and output_value_dim as head_dim
            setattr(self, f"head_{i}", SingleHeadAttention(
                input_dim=input_dim,
                output_key_query_dim=head_dim,
                output_value_dim=head_dim,
                dropout=dropout
            ))

        # Final projection layer (set bias=True per docstring)
        self.out = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Multi Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #
        # 1. Run each head independently and collect outputs
        # Each head output shape: (batch_size, num_tokens, head_dim)
        head_outputs = [getattr(self, f"head_{i}")(x) for i in range(self.num_heads)]
        
        # 2. Concatenate head outputs along the feature dimension
        # Result shape: (batch_size, num_tokens, input_dim)
        out = torch.cat(head_outputs, dim=-1)
        
        # 3. Final projection and dropout
        out = self.out(out)
        out = self.dropout(out)
        
        return out

        # ========= TODO : END ========= #


class FeedForwardLayer(nn.Module):
    """
    Class definition for Feed Forward Layer.
    """

    def __init__(self, input_dim, feedforward_dim=None, dropout=0.1):
        """
        Initialize the Feed Forward Layer.

        The model should have the following layers:
        1. A linear layer for the feedforward network. (self.fc1) **set bias to True**
        2. A GELU activation function. (self.activation)
        3. A linear layer for the feedforward network. (self.fc2) ** set bias to True**
        4. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = input_dim * 4

        # ========= TODO : START ========= #

        self.fc1 = nn.Linear(input_dim, feedforward_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(feedforward_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Feed Forward Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        ### ========= TODO : START ========= ###

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

        ### ========= TODO : END ========= ###


class LayerNorm(nn.Module):
    """
    LayerNorm module as in the paper https://arxiv.org/abs/1607.06450

    Note : Variance computation is done with biased variance.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(tuple(self.normalized_shape)))
            self.beta = nn.Parameter(torch.zeros(tuple(self.normalized_shape)))

    def forward(self, input):
        """
        Forward pass of the LayerNorm Layer.

        Args:
        input : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        normalized_input = (input - mean) / torch.sqrt(var + self.eps)  
        if self.elementwise_affine:
            normalized_input = normalized_input * self.gamma + self.beta
        return normalized_input

        # ========= TODO : END ========= #


class TransformerLayer(nn.Module):
    """
    Class definition for a single transformer layer.
    """

    def __init__(self, input_dim, num_heads, feedforward_dim=None):
        super().__init__()
        """
        Initialize the Transformer Layer.
        We will use prenorm layer where we normalize the input before applying the attention and feedforward layers.

        The model should have the following layers:
        1. A LayerNorm layer. (self.norm1)
        2. A MultiHeadAttention layer. (self.attention)
        3. A LayerNorm layer. (self.norm2)
        4. A FeedForwardLayer layer. (self.feedforward)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        # ========= TODO : START ========= #

        self.norm1 = LayerNorm(input_dim)
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.norm2 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim, input_dim * 4)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Transformer Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        # Prenorm: Normalize before attention and feedforward
        x = x + self.attention(self.norm1(x))  # Attention with residual connection
        x = x + self.feedforward(self.norm2(x))  # Feedforward with residual connection 
        return x

        # ========= TODO : END ========= #


class MiniGPT(nn.Module):
    """
    Putting it all together: GPT model
    """

    def __init__(self, config) -> None:
        super().__init__()
        """
        Putting it all together: our own GPT model!

        Initialize the MiniGPT model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.vocab_embedding)
        2. A positional embedding layer. (self.positional_embedding) We will use learnt positional embeddings. 
        3. A dropout layer for embeddings. (self.embed_dropout)
        4. Multiple TransformerLayer layers. (self.transformer_layers)
        5. A LayerNorm layer before the final layer. (self.prehead_norm)
        6. Final language Modelling head layer. (self.head) We will use weight tying (https://paperswithcode.com/method/weight-tying) and set the weights of the head layer to be the same as the vocab_embedding layer.

        NOTE: You do not need to modify anything here.
        """

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embed_dim
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )

        # prehead layer norm
        self.prehead_norm = LayerNorm(config.embed_dim)

        self.head = nn.Linear(
            config.embed_dim, config.vocab_size
        )  # Language modelling head

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the MiniGPT model.

        Remember to add the positional embeddings to your input token!!

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, seq_len) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, seq_len, vocab_size) containing the logits.
        """

        ### ========= TODO : START ========= ###

        batch_size, seq_len = x.shape
        
        # 1. Create Token and Positional Embeddings
        # tok_emb shape: (Batch, Seq_Len, Embed_Dim)
        tok_emb = self.vocab_embedding(x) 
        
        # Use the pre-registered 'pos' buffer for positional indices
        # pos_emb shape: (Seq_Len, Embed_Dim)
        pos_emb = self.positional_embedding(self.pos[:seq_len]) 
        
        # 2. Combine embeddings and apply dropout
        x = self.embed_dropout(tok_emb + pos_emb)
        
        # 3. Pass through the stack of Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # 4. Apply final layer normalization and the LM head
        x = self.prehead_norm(x)
        logits = self.head(x) # (Batch, Seq_Len, Vocab_Size)
        
        return logits

        ### ========= TODO : END ========= ###

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # GPT-2 style FFN init
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.

        Please copy the generate function from the BigramLanguageModel class you had implemented earlier.
        """

        ### ========= TODO : START ========= ###
        if not torch.is_tensor(context):
            generated = torch.tensor([context], dtype=torch.long, device=next(self.parameters()).device)
        else:
            generated = context.clone().detach().to(next(self.parameters()).device)
            if generated.dim() == 1:
                generated = generated.unsqueeze(0)

        # 2. Generation Loop
        for _ in range(max_new_tokens):
            # CROP THE CONTEXT: 
            # We must not exceed the context_length defined during initialization.
            # We access it via the model's own config attribute or the block_size.
            # Assuming you have self.positional_embedding, we can use its weight size.
            max_len = self.positional_embedding.weight.shape[0]
            context_cond = generated[:, -max_len:]
            
            # Forward pass 
            logits = self.forward(context_cond)
            
            # Focus only on the last time step: (B, T, V) -> (B, V)
            logits = logits[:, -1, :]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            generated = torch.cat((generated, next_token), dim=1)
            
        return generated[0]

        ### ========= TODO : END ========= ###
