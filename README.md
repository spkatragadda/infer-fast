# CPU tools: train, infer, finetune
make &&

# GPU trainer (nvcc + cuBLAS, sm_75)
make gpu &&

make gpu builds model.o bpe.o json.o dataset.o with g++ first, then has nvcc compile train_cuda.cu and
link them together with -lcublas.

Train on the GPU, infer on the CPU:

# 1. Pretrain from scratch on a text corpus (one string per line)
./train_cuda sample.txt pretrained.bin --epochs 400 --dim 32 --loops 4

# 2. Instruction-tune that checkpoint on your conversation JSON
./train_cuda instructions.json tuned.bin --init pretrained.bin --epochs 200

# 3. Inference — CPU only, no CUDA linked into this binary
./infer tuned.bin "User: what did the cat do Assistant:"
