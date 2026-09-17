#include "model.h"

#include "io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

// Checkpoint header, so a stale or unrelated file fails loudly
const int32_t CHECKPOINT_MAGIC = 0x49465354; // "IFST"
const int32_t CHECKPOINT_VERSION = 4; // v2 loop count, v3 residual, v4 layer norm weights

void write_matrix(std::ostream& out, const Matrix& M) {
    write_i32(out, M.rows);
    write_i32(out, M.cols);
    write_floats(out, M.data);
}

void read_matrix(std::istream& in, Matrix& M) {
    int32_t rows = read_i32(in);
    int32_t cols = read_i32(in);
    if (rows != M.rows || cols != M.cols) {
        throw std::runtime_error("checkpoint: matrix shape does not match the model");
    }
    read_floats(in, M.data);
    if (M.data.size() != static_cast<size_t>(rows) * cols) {
        throw std::runtime_error("checkpoint: matrix element count does not match its shape");
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

Matrix matmul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);

    // i-k-j order: the inner loop walks a row of B and a row of C contiguously
    for (int i = 0; i < A.rows; ++i) {
        const float* a_row = A.row(i);
        float* c_row = C.row(i);
        for (int k = 0; k < A.cols; ++k) {
            float a = a_row[k];
            const float* b_row = B.row(k);
            for (int j = 0; j < B.cols; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
    return C;
}

Matrix matmul_a_bt(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.rows);
    for (int i = 0; i < A.rows; ++i) {
        const float* a_row = A.row(i);
        float* c_row = C.row(i);
        for (int j = 0; j < B.rows; ++j) {
            const float* b_row = B.row(j);
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += a_row[k] * b_row[k];
            }
            c_row[j] = sum;
        }
    }
    return C;
}

Matrix matmul_at_b(const Matrix& A, const Matrix& B) {
    Matrix C(A.cols, B.cols);
    for (int k = 0; k < A.rows; ++k) {
        const float* a_row = A.row(k);
        const float* b_row = B.row(k);
        for (int i = 0; i < A.cols; ++i) {
            float a = a_row[i];
            float* c_row = C.row(i);
            for (int j = 0; j < B.cols; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
    return C;
}

void add_into(Matrix& dst, const Matrix& src) {
    for (size_t i = 0; i < dst.data.size(); ++i) {
        dst.data[i] += src.data[i];
    }
}

void apply_softmax(Matrix& A) {
    for (int i = 0; i < A.rows; ++i) {
        float* row = A.row(i);
        float max_val = *std::max_element(row, row + A.cols);
        float sum_exp = 0.0f;
        for (int j = 0; j < A.cols; ++j) {
            row[j] = std::exp(row[j] - max_val); // Subtract max for numerical stability
            sum_exp += row[j];
        }
        for (int j = 0; j < A.cols; ++j) {
            row[j] /= sum_exp;
        }
    }
}

void print_matrix(const Matrix& M) {
    for (int i = 0; i < M.rows; ++i) {
        const float* row = M.row(i);
        for (int j = 0; j < M.cols; ++j) {
            std::cout << row[j] << "\t";
        }
        std::cout << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

Embedding::Embedding(int vocabulary_size, int embedding_dim)
    : vocab_size(vocabulary_size), embed_dim(embedding_dim) {
    // Initialize the lookup table with basic random values (for demonstration)
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    table = Matrix(vocab_size, embed_dim);
    for (float& val : table.data) {
        val = dist(gen);
    }
    grad = Matrix(vocab_size, embed_dim);
}

const float* Embedding::lookup(int token_id) const {
    if (token_id < 0 || token_id >= vocab_size) {
        throw std::out_of_range("Embedding: token id out of range");
    }
    return table.row(token_id);
}

Matrix Embedding::forward(const std::vector<int>& token_ids) const {
    Matrix X(static_cast<int>(token_ids.size()), embed_dim);
    for (int i = 0; i < X.rows; ++i) {
        const float* src = lookup(token_ids[i]);
        std::copy(src, src + embed_dim, X.row(i));
    }
    return X;
}

void Embedding::add_positional_encoding(Matrix& X) const {
    // The encoding is a constant, so it contributes nothing to the gradient
    for (int pos = 0; pos < X.rows; ++pos) {
        float* row = X.row(pos);
        for (int j = 0; j < embed_dim; ++j) {
            float exponent = static_cast<float>(2 * (j / 2)) / static_cast<float>(embed_dim);
            float angle = static_cast<float>(pos) / std::pow(10000.0f, exponent);
            row[j] += (j % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
}

void Embedding::backward(const std::vector<int>& token_ids, const Matrix& dX) {
    for (int i = 0; i < dX.rows; ++i) {
        float* g = grad.row(token_ids[i]);
        const float* d = dX.row(i);
        for (int j = 0; j < embed_dim; ++j) {
            g[j] += d[j];
        }
    }
}

void Embedding::zero_grad() { std::fill(grad.data.begin(), grad.data.end(), 0.0f); }

void Embedding::apply_gradients(float learning_rate) {
    for (size_t i = 0; i < table.data.size(); ++i) {
        table.data[i] -= learning_rate * grad.data[i];
    }
}

float Embedding::grad_norm_sq() const {
    float total = 0.0f;
    for (float val : grad.data) {
        total += val * val;
    }
    return total;
}

void Embedding::scale_gradients(float factor) {
    for (float& val : grad.data) {
        val *= factor;
    }
}

std::vector<ParamRef> Embedding::parameters() {
    return {{"embedding.table", table.data.data(), table.rows, table.cols}};
}

void Embedding::save_weights(std::ostream& out) const { write_matrix(out, table); }

void Embedding::load_weights(std::istream& in) { read_matrix(in, table); }

// ---------------------------------------------------------------------------
// SelfAttention
// ---------------------------------------------------------------------------

SelfAttention::SelfAttention(int embedding_dim, int dimension_k, bool causal_masking)
    : embed_dim(embedding_dim), head_dim(dimension_k), causal(causal_masking) {
    // Initialize weights with basic random values (for demonstration)
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto init_matrix = [&](int r, int c) {
        Matrix M(r, c);
        for (float& val : M.data)
            val = dist(gen);
        return M;
    };

    Wq = init_matrix(embed_dim, head_dim);
    Wk = init_matrix(embed_dim, head_dim);
    Wv = init_matrix(embed_dim, head_dim);

    dWq = Matrix(embed_dim, head_dim);
    dWk = Matrix(embed_dim, head_dim);
    dWv = Matrix(embed_dim, head_dim);
}

Matrix SelfAttention::forward(const Matrix& X) {
    if (X.cols != embed_dim) {
        throw std::runtime_error("SelfAttention: input width does not match embed_dim");
    }

    Cache cache;
    cache.X = X;

    // 1. Calculate Query, Key, and Value matrices
    cache.Q = matmul(X, Wq);
    cache.K = matmul(X, Wk);
    cache.V = matmul(X, Wv);

    // 2. Calculate Attention Scores: (Q * K^T)
    Matrix scores = matmul_a_bt(cache.Q, cache.K);

    // 3. Scale the scores by sqrt(d_k)
    float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (float& val : scores.data) {
        val *= inv_scale;
    }

    // 4. Mask future positions, then apply Softmax to get attention weights
    if (causal) {
        for (int i = 0; i < scores.rows; ++i) {
            for (int j = i + 1; j < scores.cols; ++j) {
                scores(i, j) = -std::numeric_limits<float>::infinity();
            }
        }
    }
    apply_softmax(scores);
    cache.A = scores;

    // 5. Multiply weights by Values: scores * V
    Matrix output = matmul(cache.A, cache.V);
    cache_stack.push_back(std::move(cache));
    return output;
}

Matrix SelfAttention::backward(const Matrix& dOut) {
    if (cache_stack.empty()) {
        throw std::runtime_error("SelfAttention: backward without a matching forward");
    }

    // Pop the newest iteration: the loop is unwound last-in first-out
    Cache cache = std::move(cache_stack.back());
    cache_stack.pop_back();

    // 5. output = A * V
    Matrix dA = matmul_a_bt(dOut, cache.V);
    Matrix dV = matmul_at_b(cache.A, dOut);

    // 4. Softmax backward, row by row: dS = A * (dA - sum(dA * A))
    //    Masked entries hold A = 0, so their gradient falls out to zero on its own.
    Matrix dScores(cache.A.rows, cache.A.cols);
    for (int i = 0; i < cache.A.rows; ++i) {
        const float* a = cache.A.row(i);
        const float* da = dA.row(i);
        float* ds = dScores.row(i);
        float dot = 0.0f;
        for (int j = 0; j < cache.A.cols; ++j) {
            dot += da[j] * a[j];
        }
        for (int j = 0; j < cache.A.cols; ++j) {
            ds[j] = a[j] * (da[j] - dot);
        }
    }

    // 3. Undo the scaling
    float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (float& val : dScores.data) {
        val *= inv_scale;
    }

    // 2. scores = Q * K^T
    Matrix dQ = matmul(dScores, cache.K);
    Matrix dK = matmul_at_b(dScores, cache.Q);

    // 1. Q, K, V = X * W  ->  dW = X^T * dW_out. Every iteration shares one set of
    //    weights, so these accumulate across the whole loop before a single update.
    add_into(dWq, matmul_at_b(cache.X, dQ));
    add_into(dWk, matmul_at_b(cache.X, dK));
    add_into(dWv, matmul_at_b(cache.X, dV));

    // dX collects all three paths, and becomes dOut for the previous iteration
    Matrix dX = matmul_a_bt(dQ, Wq);
    add_into(dX, matmul_a_bt(dK, Wk));
    add_into(dX, matmul_a_bt(dV, Wv));
    return dX;
}

void SelfAttention::clear_cache() { cache_stack.clear(); }

void SelfAttention::zero_grad() {
    std::fill(dWq.data.begin(), dWq.data.end(), 0.0f);
    std::fill(dWk.data.begin(), dWk.data.end(), 0.0f);
    std::fill(dWv.data.begin(), dWv.data.end(), 0.0f);
}

void SelfAttention::apply_gradients(float learning_rate) {
    auto step = [&](Matrix& W, const Matrix& dW) {
        for (size_t i = 0; i < W.data.size(); ++i) {
            W.data[i] -= learning_rate * dW.data[i];
        }
    };
    step(Wq, dWq);
    step(Wk, dWk);
    step(Wv, dWv);
}

float SelfAttention::grad_norm_sq() const {
    float total = 0.0f;
    auto accumulate = [&](const Matrix& dW) {
        for (float val : dW.data) {
            total += val * val;
        }
    };
    accumulate(dWq);
    accumulate(dWk);
    accumulate(dWv);
    return total;
}

void SelfAttention::scale_gradients(float factor) {
    auto scale = [&](Matrix& dW) {
        for (float& val : dW.data) {
            val *= factor;
        }
    };
    scale(dWq);
    scale(dWk);
    scale(dWv);
}

std::vector<ParamRef> SelfAttention::parameters() {
    return {{"attention.Wq", Wq.data.data(), Wq.rows, Wq.cols},
            {"attention.Wk", Wk.data.data(), Wk.rows, Wk.cols},
            {"attention.Wv", Wv.data.data(), Wv.rows, Wv.cols}};
}

void SelfAttention::save_weights(std::ostream& out) const {
    write_matrix(out, Wq);
    write_matrix(out, Wk);
    write_matrix(out, Wv);
}

void SelfAttention::load_weights(std::istream& in) {
    read_matrix(in, Wq);
    read_matrix(in, Wk);
    read_matrix(in, Wv);
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

LayerNorm::LayerNorm(int feature_dim) : dim(feature_dim) {
    // Start as the identity: unit scale, zero shift
    gamma.assign(dim, 1.0f);
    beta.assign(dim, 0.0f);
    dgamma.assign(dim, 0.0f);
    dbeta.assign(dim, 0.0f);
}

Matrix LayerNorm::forward(const Matrix& X) {
    if (X.cols != dim) {
        throw std::runtime_error("LayerNorm: input width does not match feature_dim");
    }

    const float epsilon = 1e-5f;
    Cache cache;
    cache.normalized = Matrix(X.rows, X.cols);
    cache.inv_std.resize(static_cast<size_t>(X.rows));

    Matrix Y(X.rows, X.cols);
    for (int i = 0; i < X.rows; ++i) {
        const float* x = X.row(i);

        // Mean and variance across this token's features
        float mean = 0.0f;
        for (int j = 0; j < dim; ++j) {
            mean += x[j];
        }
        mean /= static_cast<float>(dim);

        float variance = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = x[j] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(dim);

        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        cache.inv_std[i] = inv_std;

        float* norm_row = cache.normalized.row(i);
        float* y = Y.row(i);
        for (int j = 0; j < dim; ++j) {
            norm_row[j] = (x[j] - mean) * inv_std;
            y[j] = gamma[j] * norm_row[j] + beta[j];
        }
    }

    cache_stack.push_back(std::move(cache));
    return Y;
}

Matrix LayerNorm::backward(const Matrix& dY) {
    if (cache_stack.empty()) {
        throw std::runtime_error("LayerNorm: backward without a matching forward");
    }

    Cache cache = std::move(cache_stack.back());
    cache_stack.pop_back();

    Matrix dX(dY.rows, dY.cols);
    for (int i = 0; i < dY.rows; ++i) {
        const float* dy = dY.row(i);
        const float* norm_row = cache.normalized.row(i);
        float* dx = dX.row(i);

        // Scale and shift are per-feature, so their gradients sum over tokens
        float mean_dnorm = 0.0f;
        float mean_dnorm_norm = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float dnorm = dy[j] * gamma[j];
            dgamma[j] += dy[j] * norm_row[j];
            dbeta[j] += dy[j];
            mean_dnorm += dnorm;
            mean_dnorm_norm += dnorm * norm_row[j];
        }
        mean_dnorm /= static_cast<float>(dim);
        mean_dnorm_norm /= static_cast<float>(dim);

        // Subtracting both means is what routes the gradient through the mean
        // and variance the normalization itself computed
        for (int j = 0; j < dim; ++j) {
            float dnorm = dy[j] * gamma[j];
            dx[j] = cache.inv_std[i] * (dnorm - mean_dnorm - norm_row[j] * mean_dnorm_norm);
        }
    }
    return dX;
}

void LayerNorm::clear_cache() { cache_stack.clear(); }

void LayerNorm::zero_grad() {
    std::fill(dgamma.begin(), dgamma.end(), 0.0f);
    std::fill(dbeta.begin(), dbeta.end(), 0.0f);
}

void LayerNorm::apply_gradients(float learning_rate) {
    for (int j = 0; j < dim; ++j) {
        gamma[j] -= learning_rate * dgamma[j];
        beta[j] -= learning_rate * dbeta[j];
    }
}

float LayerNorm::grad_norm_sq() const {
    float total = 0.0f;
    for (int j = 0; j < dim; ++j) {
        total += dgamma[j] * dgamma[j] + dbeta[j] * dbeta[j];
    }
    return total;
}

void LayerNorm::scale_gradients(float factor) {
    for (int j = 0; j < dim; ++j) {
        dgamma[j] *= factor;
        dbeta[j] *= factor;
    }
}

std::vector<ParamRef> LayerNorm::parameters() {
    return {{"norm.gamma", gamma.data(), dim, 1},
            {"norm.beta", beta.data(), dim, 1}};
}

void LayerNorm::save_weights(std::ostream& out) const {
    write_floats(out, gamma);
    write_floats(out, beta);
}

void LayerNorm::load_weights(std::istream& in) {
    read_floats(in, gamma);
    read_floats(in, beta);
    if (gamma.size() != static_cast<size_t>(dim) || beta.size() != static_cast<size_t>(dim)) {
        throw std::runtime_error("checkpoint: layer norm size does not match the model");
    }
}

// ---------------------------------------------------------------------------
// OutputHead
// ---------------------------------------------------------------------------

OutputHead::OutputHead(int input_dim, int vocabulary_size)
    : in_dim(input_dim), vocab_size(vocabulary_size) {
    std::mt19937 gen(13);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    W = Matrix(in_dim, vocab_size);
    for (float& val : W.data) {
        val = dist(gen);
    }
    b.assign(vocab_size, 0.0f);
    dW = Matrix(in_dim, vocab_size);
    db.assign(vocab_size, 0.0f);
}

Matrix OutputHead::forward(const Matrix& H) {
    in_cache = H;

    Matrix logits = matmul(H, W);
    for (int i = 0; i < logits.rows; ++i) {
        float* row = logits.row(i);
        for (int j = 0; j < vocab_size; ++j) {
            row[j] += b[j];
        }
    }

    apply_softmax(logits);
    probs_cache = logits;
    return probs_cache;
}

float OutputHead::loss_impl(const std::vector<int>& targets,
                            const std::vector<char>* mask) const {
    const float epsilon = 1e-9f;
    float total = 0.0f;
    int counted = 0;
    for (int i = 0; i < probs_cache.rows; ++i) {
        if (mask && !(*mask)[i]) continue;
        total -= std::log(probs_cache(i, targets[i]) + epsilon);
        ++counted;
    }
    if (counted == 0) return 0.0f;
    return total / static_cast<float>(counted);
}

Matrix OutputHead::backward_impl(const std::vector<int>& targets,
                                 const std::vector<char>* mask) {
    int seq_length = probs_cache.rows;

    // Average over the positions that actually count, so a mostly-masked
    // sequence still produces a full-sized gradient for the few it trains on
    int counted = 0;
    for (int i = 0; i < seq_length; ++i) {
        if (!mask || (*mask)[i]) ++counted;
    }

    Matrix dLogits(seq_length, vocab_size);
    if (counted > 0) {
        float inv_counted = 1.0f / static_cast<float>(counted);
        for (int i = 0; i < seq_length; ++i) {
            if (mask && !(*mask)[i]) continue; // Masked rows stay zero

            const float* probs = probs_cache.row(i);
            float* row = dLogits.row(i);
            for (int j = 0; j < vocab_size; ++j) {
                row[j] = probs[j] * inv_counted;
            }
            row[targets[i]] -= inv_counted;
        }
    }

    add_into(dW, matmul_at_b(in_cache, dLogits));
    for (int i = 0; i < seq_length; ++i) {
        const float* row = dLogits.row(i);
        for (int j = 0; j < vocab_size; ++j) {
            db[j] += row[j];
        }
    }

    return matmul_a_bt(dLogits, W);
}

float OutputHead::loss(const std::vector<int>& targets) const {
    return loss_impl(targets, nullptr);
}

float OutputHead::loss(const std::vector<int>& targets, const std::vector<char>& mask) const {
    return loss_impl(targets, &mask);
}

Matrix OutputHead::backward(const std::vector<int>& targets) {
    return backward_impl(targets, nullptr);
}

Matrix OutputHead::backward(const std::vector<int>& targets, const std::vector<char>& mask) {
    return backward_impl(targets, &mask);
}

void OutputHead::zero_grad() {
    std::fill(dW.data.begin(), dW.data.end(), 0.0f);
    std::fill(db.begin(), db.end(), 0.0f);
}

void OutputHead::apply_gradients(float learning_rate) {
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] -= learning_rate * dW.data[i];
    }
    for (size_t j = 0; j < b.size(); ++j) {
        b[j] -= learning_rate * db[j];
    }
}

float OutputHead::grad_norm_sq() const {
    float total = 0.0f;
    for (float val : dW.data) {
        total += val * val;
    }
    for (float val : db) {
        total += val * val;
    }
    return total;
}

void OutputHead::scale_gradients(float factor) {
    for (float& val : dW.data) {
        val *= factor;
    }
    for (float& val : db) {
        val *= factor;
    }
}

std::vector<ParamRef> OutputHead::parameters() {
    return {{"head.W", W.data.data(), W.rows, W.cols},
            {"head.b", b.data(), vocab_size, 1}};
}

void OutputHead::save_weights(std::ostream& out) const {
    write_matrix(out, W);
    write_floats(out, b);
}

void OutputHead::load_weights(std::istream& in) {
    read_matrix(in, W);
    read_floats(in, b);
    if (b.size() != static_cast<size_t>(vocab_size)) {
        throw std::runtime_error("checkpoint: output bias size does not match the vocabulary");
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

Model::Model(const ModelConfig& cfg, BPETokenizer trained_tokenizer)
    : config(cfg),
      tokenizer(std::move(trained_tokenizer)),
      embedding(cfg.vocab_size, cfg.embed_dim),
      attention(cfg.embed_dim, cfg.head_dim, cfg.causal),
      norm(cfg.embed_dim),
      head(cfg.head_dim, cfg.vocab_size) {
    if (config.num_loops < 1) {
        throw std::runtime_error("Model: num_loops must be at least 1");
    }
    // Each iteration feeds its output back in as the next iteration's input,
    // so the block has to be square once it runs more than once.
    if (config.num_loops > 1 && config.head_dim != config.embed_dim) {
        throw std::runtime_error("Model: a looped attention block needs head_dim == embed_dim");
    }
}

Matrix Model::forward(const std::vector<int>& token_ids) {
    attention.clear_cache();
    norm.clear_cache();

    // Embed once, positions included
    Matrix X = embedding.forward(token_ids);
    embedding.add_positional_encoding(X);

    // Then refine that representation by running the same block num_loops times:
    //   x_next = Norm(x + g(x))
    // The residual keeps a direct path back to the embedding; the norm holds the
    // scale steady so repeated iterations cannot let it drift.
    for (int loop = 0; loop < config.num_loops; ++loop) {
        Matrix residual = attention.forward(X);
        add_into(residual, X);          // x + g(x)
        X = norm.forward(residual);     // Norm(x + g(x))
    }

    // Only the final iteration reaches the head
    return head.forward(X);
}

float Model::clip_gradients(float max_norm) {
    float sum_sq = embedding.grad_norm_sq() + attention.grad_norm_sq() +
                   norm.grad_norm_sq() + head.grad_norm_sq();
    float total = std::sqrt(sum_sq);

    auto rescale = [&](float factor) {
        embedding.scale_gradients(factor);
        attention.scale_gradients(factor);
        norm.scale_gradients(factor);
        head.scale_gradients(factor);
    };

    if (!std::isfinite(total)) {
        // An overflowed gradient would poison every weight with NaN on the next
        // update, and nothing recovers from that. Drop the step instead.
        rescale(0.0f);
        return total;
    }

    if (max_norm > 0.0f && total > max_norm) {
        rescale(max_norm / (total + 1e-6f));
    }
    return total;
}

float Model::train_step(const std::vector<int>& sequence, float learning_rate,
                        float max_grad_norm) {
    return train_step_impl(sequence, nullptr, learning_rate, max_grad_norm);
}

float Model::train_step_masked(const std::vector<int>& sequence,
                               const std::vector<char>& train_on, float learning_rate,
                               float max_grad_norm) {
    if (train_on.size() != sequence.size()) {
        throw std::runtime_error("train_step_masked: mask length does not match the sequence");
    }
    return train_step_impl(sequence, &train_on, learning_rate, max_grad_norm);
}

float Model::train_step_impl(const std::vector<int>& sequence,
                             const std::vector<char>* train_on, float learning_rate,
                             float max_grad_norm) {
    if (sequence.size() < 2) return 0.0f;

    // Predict each token from the ones before it
    std::vector<int> inputs(sequence.begin(), sequence.end() - 1);
    std::vector<int> targets(sequence.begin() + 1, sequence.end());

    // The mask marks tokens to produce, so it shifts with the targets: position i
    // predicts sequence[i + 1], and is trained only if that token is flagged.
    std::vector<char> target_mask;
    if (train_on) {
        target_mask.assign(train_on->begin() + 1, train_on->end());
    }

    embedding.zero_grad();
    attention.zero_grad();
    norm.zero_grad();
    head.zero_grad();

    forward(inputs);
    // Taken once, on the final iteration's output
    float loss = train_on ? head.loss(targets, target_mask) : head.loss(targets);

    // Unroll the loop in reverse: each backward() pops one iteration's cache and
    // hands the gradient to the iteration before it, accumulating into the shared
    // weights the whole way down. Per iteration that means undoing the norm, then
    // splitting the residual: the gradient reaching x + g(x) flows both into the
    // block and straight around it.
    Matrix dX = train_on ? head.backward(targets, target_mask) : head.backward(targets);
    for (int loop = 0; loop < config.num_loops; ++loop) {
        dX = norm.backward(dX);
        Matrix dBlock = attention.backward(dX);
        add_into(dBlock, dX); // The skip path carries the gradient through unchanged
        dX = std::move(dBlock);
    }
    embedding.backward(inputs, dX);

    // Bound the whole step before it lands, so one outlier sequence cannot
    // throw the weights somewhere they never come back from
    clip_gradients(max_grad_norm);

    // One update, after the full unroll
    head.apply_gradients(learning_rate);
    norm.apply_gradients(learning_rate);
    attention.apply_gradients(learning_rate);
    embedding.apply_gradients(learning_rate);

    return loss;
}

std::vector<int> Model::generate(const std::vector<int>& prompt_ids, int num_tokens) {
    std::vector<int> ids = prompt_ids;
    for (int step = 0; step < num_tokens; ++step) {
        // Each forward() runs the full attention loop before a token comes out
        Matrix probs = forward(ids);

        // Causal masking means the last row is the distribution for the next token
        const float* last = probs.row(probs.rows - 1);
        int best = static_cast<int>(std::max_element(last, last + probs.cols) - last);
        ids.push_back(best);
    }
    return ids;
}

std::vector<ParamRef> Model::parameters() {
    std::vector<ParamRef> all;
    for (const auto& block : {embedding.parameters(), attention.parameters(),
                              norm.parameters(), head.parameters()}) {
        all.insert(all.end(), block.begin(), block.end());
    }
    return all;
}

void Model::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not open '" + path + "' for writing");
    }

    write_i32(out, CHECKPOINT_MAGIC);
    write_i32(out, CHECKPOINT_VERSION);
    write_i32(out, config.embed_dim);
    write_i32(out, config.head_dim);
    write_i32(out, config.vocab_size);
    write_i32(out, config.num_loops);
    write_i32(out, config.causal ? 1 : 0);

    tokenizer.save(out);
    embedding.save_weights(out);
    attention.save_weights(out);
    norm.save_weights(out);
    head.save_weights(out);

    if (!out) {
        throw std::runtime_error("failed while writing '" + path + "'");
    }
}

Model Model::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not open '" + path + "' for reading");
    }

    if (read_i32(in) != CHECKPOINT_MAGIC) {
        throw std::runtime_error("'" + path + "' is not an infer-fast checkpoint");
    }
    int32_t version = read_i32(in);
    if (version != CHECKPOINT_VERSION) {
        throw std::runtime_error("checkpoint version " + std::to_string(version) +
                                 " is not supported by this build");
    }

    ModelConfig cfg;
    cfg.embed_dim = read_i32(in);
    cfg.head_dim = read_i32(in);
    cfg.vocab_size = read_i32(in);
    cfg.num_loops = read_i32(in);
    cfg.causal = read_i32(in) != 0;

    BPETokenizer tokenizer;
    tokenizer.load(in);
    if (tokenizer.vocab_size() != cfg.vocab_size) {
        throw std::runtime_error("checkpoint: tokenizer vocabulary disagrees with the model");
    }

    Model model(cfg, std::move(tokenizer));
    model.embedding.load_weights(in);
    model.attention.load_weights(in);
    model.norm.load_weights(in);
    model.head.load_weights(in);
    return model;
}
