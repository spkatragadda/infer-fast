#ifndef MODEL_H
#define MODEL_H

#include <iosfwd>
#include <string>
#include <vector>

#include "bpe.h"

// A 2D matrix held as one contiguous row-major buffer:
// element (i, j) lives at data[i * cols + j]
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(int r, int c, float fill = 0.0f)
        : rows(r), cols(c), data(static_cast<size_t>(r) * c, fill) {}

    float& operator()(int i, int j) { return data[static_cast<size_t>(i) * cols + j]; }
    float operator()(int i, int j) const { return data[static_cast<size_t>(i) * cols + j]; }

    // Pointer to the start of row i, so callers can walk a row contiguously
    float* row(int i) { return data.data() + static_cast<size_t>(i) * cols; }
    const float* row(int i) const { return data.data() + static_cast<size_t>(i) * cols; }
};

// A handle on one weight tensor, so another backend (the CUDA trainer) can copy
// weights in and out without duplicating the checkpoint format
struct ParamRef {
    const char* name;
    float* data;
    int rows;
    int cols; // 1 for vectors

    size_t size() const { return static_cast<size_t>(rows) * cols; }
};

// ---------------------------------------------------------------------------
// Matrix helpers, shared by the forward and backward passes
// ---------------------------------------------------------------------------

Matrix matmul(const Matrix& A, const Matrix& B);       // A * B
Matrix matmul_a_bt(const Matrix& A, const Matrix& B);  // A * B^T
Matrix matmul_at_b(const Matrix& A, const Matrix& B);  // A^T * B
void add_into(Matrix& dst, const Matrix& src);
void apply_softmax(Matrix& A);
void print_matrix(const Matrix& M);

// ---------------------------------------------------------------------------
// Embedding table
// ---------------------------------------------------------------------------

class Embedding {
private:
    int vocab_size;
    int embed_dim;
    Matrix table; // One row per token in the vocabulary
    Matrix grad;  // Same shape as the table

public:
    Embedding(int vocabulary_size, int embedding_dim);

    // Look up a single token id, returning a pointer to its row
    const float* lookup(int token_id) const;

    // Forward Pass: token ids -> (Seq_Length x Embed_Dim) matrix
    Matrix forward(const std::vector<int>& token_ids) const;

    // Add sinusoidal positional information so order matters to the attention
    void add_positional_encoding(Matrix& X) const;

    // Backward Pass: route each row of dX back to the token that produced it
    void backward(const std::vector<int>& token_ids, const Matrix& dX);

    void zero_grad();
    void apply_gradients(float learning_rate);

    // Sum of squared gradients, and a uniform rescale: together these let the
    // model clip every parameter against one global norm
    float grad_norm_sq() const;
    void scale_gradients(float factor);

    std::vector<ParamRef> parameters();

    void save_weights(std::ostream& out) const;
    void load_weights(std::istream& in);

    int dim() const { return embed_dim; }
};

// ---------------------------------------------------------------------------
// Single-head self-attention
// ---------------------------------------------------------------------------

class SelfAttention {
private:
    int embed_dim;
    int head_dim;
    bool causal; // Mask out future tokens, so next-token training cannot peek ahead
    Matrix Wq, Wk, Wv;
    Matrix dWq, dWk, dWv;

    // Activations one forward pass needs to hand to its backward pass
    struct Cache {
        Matrix X, Q, K, V, A;
    };

    // The block is applied in a loop, so caches stack up: forward() pushes one
    // per iteration and backward() pops them in reverse order.
    std::vector<Cache> cache_stack;

public:
    SelfAttention(int embedding_dim, int dimension_k, bool causal_masking = true);

    Matrix forward(const Matrix& X);

    // Backward Pass: takes dLoss/dOutput for the most recent unconsumed forward,
    // accumulates weight gradients, and returns dLoss/dX for that iteration
    Matrix backward(const Matrix& dOut);

    // Drop caches left over from a forward pass that was never backpropagated
    void clear_cache();

    void zero_grad();
    void apply_gradients(float learning_rate);

    float grad_norm_sq() const;
    void scale_gradients(float factor);

    std::vector<ParamRef> parameters();

    void save_weights(std::ostream& out) const;
    void load_weights(std::istream& in);
};

// ---------------------------------------------------------------------------
// Layer normalization, applied across the features of each token independently:
//   y = gamma * (x - mean) / sqrt(var + eps) + beta
// ---------------------------------------------------------------------------

class LayerNorm {
private:
    int dim;
    std::vector<float> gamma; // Learned scale, one per feature
    std::vector<float> beta;  // Learned shift, one per feature
    std::vector<float> dgamma;
    std::vector<float> dbeta;

    // What one forward pass owes its backward pass
    struct Cache {
        Matrix normalized;          // (x - mean) / sqrt(var + eps)
        std::vector<float> inv_std; // 1 / sqrt(var + eps), one per token
    };

    // Sits inside the loop like the attention block, so its caches stack too
    std::vector<Cache> cache_stack;

public:
    explicit LayerNorm(int feature_dim);

    Matrix forward(const Matrix& X);

    // Backward Pass: takes dLoss/dY for the most recent unconsumed forward,
    // accumulates gamma/beta gradients, and returns dLoss/dX
    Matrix backward(const Matrix& dY);

    void clear_cache();
    void zero_grad();
    void apply_gradients(float learning_rate);

    float grad_norm_sq() const;
    void scale_gradients(float factor);

    std::vector<ParamRef> parameters();

    void save_weights(std::ostream& out) const;
    void load_weights(std::istream& in);
};

// ---------------------------------------------------------------------------
// Output head: projects attention output to vocabulary logits, with a
// softmax + cross-entropy loss for next-token prediction
// ---------------------------------------------------------------------------

class OutputHead {
private:
    int in_dim;
    int vocab_size;
    Matrix W;                // (in_dim x vocab_size)
    std::vector<float> b;
    Matrix dW;
    std::vector<float> db;

    // Activations cached by forward() for the backward pass
    Matrix in_cache, probs_cache;

    // Shared by the plain and masked variants; a null mask counts every position
    float loss_impl(const std::vector<int>& targets, const std::vector<char>* mask) const;
    Matrix backward_impl(const std::vector<int>& targets, const std::vector<char>* mask);

public:
    OutputHead(int input_dim, int vocabulary_size);

    // Forward Pass: attention output -> per-position probability over the vocabulary
    Matrix forward(const Matrix& H);

    // Mean cross-entropy of the cached probabilities against the target tokens
    float loss(const std::vector<int>& targets) const;

    // The same, averaged over only the positions where mask is non-zero
    float loss(const std::vector<int>& targets, const std::vector<char>& mask) const;

    // Backward Pass: softmax + cross-entropy collapse to (probs - one_hot)
    Matrix backward(const std::vector<int>& targets);

    // The same, with masked-out positions contributing no gradient at all
    Matrix backward(const std::vector<int>& targets, const std::vector<char>& mask);

    void zero_grad();
    void apply_gradients(float learning_rate);

    float grad_norm_sq() const;
    void scale_gradients(float factor);

    std::vector<ParamRef> parameters();

    void save_weights(std::ostream& out) const;
    void load_weights(std::istream& in);
};

// ---------------------------------------------------------------------------
// The whole model: tokenizer + embedding + attention + head, saved as one file
// ---------------------------------------------------------------------------

struct ModelConfig {
    int embed_dim = 16;
    int head_dim = 16;
    int vocab_size = 0;
    int num_loops = 4; // Times the shared attention block is applied per forward pass
    bool causal = true;
};

class Model {
public:
    ModelConfig config;
    BPETokenizer tokenizer;
    Embedding embedding;
    SelfAttention attention;
    LayerNorm norm; // Shared across iterations, like the attention block
    OutputHead head;

    Model(const ModelConfig& cfg, BPETokenizer trained_tokenizer);

    // Forward Pass: embed once, run the shared attention block config.num_loops
    // times as X = Norm(X + block(X)), then project the final iteration's output
    // to vocabulary probabilities
    Matrix forward(const std::vector<int>& token_ids);

    // One SGD step on a single sequence, training it to predict its own next
    // token. The loss is taken once, on the final iteration's output; its
    // gradient is then unrolled back through every iteration and accumulated
    // into the shared weights, clipped, and applied in a single update.
    // Returns that loss.
    float train_step(const std::vector<int>& sequence, float learning_rate,
                     float max_grad_norm = 1.0f);

    // The same step for instruction tuning: train_on runs parallel to sequence
    // and marks the tokens the model should be taught to produce. Prompt tokens
    // are still attended to, they just contribute no loss and no gradient.
    float train_step_masked(const std::vector<int>& sequence,
                            const std::vector<char>& train_on, float learning_rate,
                            float max_grad_norm = 1.0f);

    // Scale every gradient down so their combined L2 norm is at most max_norm,
    // and return the norm measured before clipping. A non-positive max_norm
    // disables clipping.
    float clip_gradients(float max_norm);

    // Greedily continue a prompt for a number of tokens, returning the full id list
    std::vector<int> generate(const std::vector<int>& prompt_ids, int num_tokens);

    // Every weight tensor, in a fixed order. The CUDA trainer uploads these,
    // trains, writes them back, and then saves through the same code path the
    // CPU trainer uses, so one checkpoint format serves both.
    std::vector<ParamRef> parameters();

    void save(const std::string& path) const;
    static Model load(const std::string& path);

private:
    // Shared by the plain and masked steps; a null mask trains on every token
    float train_step_impl(const std::vector<int>& sequence, const std::vector<char>* train_on,
                          float learning_rate, float max_grad_norm);
};

#endif // MODEL_H
