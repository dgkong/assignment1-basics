### Problem (unicode1): Understanding Unicode

1. chr(0) == '\x00'
2. repr(chr(0)) == "'\\x00'"   
print(chr(0)) prints nothing.
3. "this is a test" + chr(0) + "string" adds the '\x00' representation into the string.   
print("this is a test" + chr(0) + "string") shows nothing between "test" and "string".

### Problem (unicode2): Unicode Encodings

1. UTF8 can fit single byte values in minimum space and flexibly incorporate more bytes when needed. UTF16 and UTF32 increase complexity and can waste space through multiple byte allocations for even ASCII values that only require a single byte.
2. Outside of the ASCII values that can be represented by a single byte, the UTF8 encoding uses a sequence of bytes to represent some character. These cannot be decoded sequentially byte by byte. Example used: "hello 가" (b'hello \xea\xb0\x80')
3. b'\xC1\xA1' - This follows the 2-byte encoding format in UTF-8 including the correct lead byte and continuation byte but encodes the same value as 'a'. This byte sequence therefore does not decode to any Unicode character as this would introduce duplicate representations.

### Problem (train_bpe_tinystories): BPE Training on TinyStories

1. ~2m and 304MB memory. Longest vocab: b' accomplishment', b' disappointment', and b' responsibility'.

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText

1. ~1h 22m and 7GB memory. Longest vocab: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'.

### Problem (tokenizer_experiments): Experiments with tokenizers

1. ts average compression ratio: 4.0883; owt average compression ratio: 4.4843.
2. owt average compression ratio: 3.1924.
3. throughput: 6.0286 MB per second tokenizing TinyStories-valid. If we extrapolate this throughput, we'd tokenize the Pile dataset in ~38 hours.
4. uint16 is fitting especially since our token ids don't require negative values. 16 bits fits from 0~65535 which is sufficient for our vocab size as well for owt and ts.

### Problem (transformer_accounting): Transformer LM resource accounting

1. 2,127,057,600 trainable parameters. 8,508,230,400 bytes ~ 7.9 GB.
2. Matmuls: 4,513,336,524,800 FLOPS

    num_layers x MultiHeadSelfAttention: ~27 billion
    - q_proj(x), k_proj(x), v_proj(x): (d_model, d_model) (d_model, context_length)
    - qk (context_length, d_model) (d_model, context_length)
    - qkv (context_length, context_length) (context_length, d_model)
    - output (d_model, d_model) (context_length, d_model)

    num_layers x SwiGLU: ~62 billion
    - w1(x), w3(x): (d_model, d_ff) (context_length, d_model)
    - w2(x): (d_ff, d_model) (context_length, d_ff)

    lm_head(x): (d_model, vocab_size) (context_length, d_model)
3. The MultiHeadSelfAttention, especially the FFN, takes the majority of the FLOPS.
4. skip
5. If context length is 16,384, FLOPs from SwiGLU should increase proportionally, but the qk and qkv matrix multiplication FLOPs would increase quadratically.

### Problem (learning_rate_tuning): Tuning the learning rate
- For this SGD example, lr of 1e1 brings loss from 20's to ~3, lr of 1e2 brings loss down to ~3e-23, and lr of 1e3 diverges.

### Problem (adamwAccounting): Resource accounting for training with AdamW
1.  batch_size (vocab_size, context_length, num_layers, d_model, num_heads) d_ff = 4 * d_model

    Parameters:

        - Transformer block: num_layers
            - RMSNorms: 2 * d_model
            - MultiheadSelfAttention: 4 * d_model * d_model
            - FFN: 2 * d_model * 4 * d_model
        - final RMSNorm: d_model
        - output embedding: d_model * vocab_size

    Optimizer state: 

        - m, v -> 2 * Parameters 

    Gradients: 
    
        - 1 * Parameters

    Activations:

        - Transformer block: num_layers
            - RMSNorms: 2 * batch_size * context_length * d_model
            - MultiheadSelfAttention:
                - QKV proj: 3 * batch_size * context_length * d_model
                - QK matmul: batch_size * num_heads * context_length * context_length
                - softmax: batch_size * num_heads * context_length * context_length
                - weighted sum of values: batch_size * num_heads * context_length * d_model
                - output projection: batch_size * context_length * d_model
            - FFN:
                - W1: batch_size * context_length * 4 * d_model
                - SiLU: batch_size * context_length * 4 * d_model
                - W2: batch_size * context_length * d_model
        - final RMSNorm: batch_size * context_length * d_model
        - output embedding: batch_size * context_length * vocab_size
        - cross entropy: batch_size * context_length
2. skip
3. Total Flops: ~6 * Parameters * Tokens (forward: 2 * P * T, backward: gradient wrt Parameters: 2 * P * T, gradient wrt Activations: 2 * P * T)
4. skip

### Problem (learning_rate): Tune the learning rate
1. For batch size 8, surprisingly divergence occurred at a high max learning rate of 2e-1 where we can clearly see signs of loss increasing rather than getting stuck at a high plateau. The best max learning rate was around 8e-4.
2. It may be because I'm training the downscaled version (way less tokens seen and optimizer steps), but the learning rate way under the edge of stability was best for bringing down losses. Perhaps the higher learning rate training runs would've converged to a lower loss if trained for more steps.

For the sake of time, will be skipping experiments/ablations due to length of training runs even on downscaled configs (~1 hour 20 minutes). 

### Problem (batch_size_experiment): Batch size variations
1. Micro-batch size of 8 gave fastest tokens per second on my device. Batch size 32 gives lower losses than batch size 8.

### Problem (generate): Generate text
1. Input prompt: "Once"

    Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. One day, he saw a new car in his room. It was red and shiny. Tim wanted to play with it, but he was scared.
    Tim's mom saw him and said, "Don't be scared, Tim. It's just a little scared. It's just a car." Tim felt better and they played together. They had fun and learned to be brave.
    The next day, Tim's mom said, "Tim, you need to be brave and find a way to play with the car." Tim was happy and promised to be brave. They played together and had a lot of fun.

### Problem (layer_norm_ablation): Remove RMSNorm and train

### Problem (pre_norm_ablation): Implement post-norm and train

### Problem (no_pos_emb): Implement NoPE

### Problem (swiglu_ablation): SwiGLU vs. SiLU
