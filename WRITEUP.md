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
