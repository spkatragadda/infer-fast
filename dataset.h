#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>

#include "bpe.h"
#include "json.h"

// One training sequence: token ids, plus a parallel flag marking the tokens the
// model is graded on producing. Pretraining grades everything; instruction
// tuning grades only the assistant's replies.
struct Example {
    std::vector<int> tokens;
    std::vector<char> train_on;
    int trainable = 0;
};

// Append tokens, recording whether the model is graded on producing them
void append_tokens(Example& example, const std::vector<int>& ids, bool trainable);

// --- Plain text: one training string per line, blank lines skipped ---
std::vector<std::string> read_text_lines(const std::string& path);
std::vector<Example> build_text_examples(const BPETokenizer& tokenizer,
                                         const std::vector<std::string>& lines);

// --- Chat JSON: an array of conversations, each an array of messages shaped
// {"role": "user" | "assistant" | "system", "content": "..."}. A flat array of
// message objects is read as a single conversation. ---
Example build_chat_example(const BPETokenizer& tokenizer, const JsonValue& conversation);
std::vector<Example> build_chat_examples(const BPETokenizer& tokenizer, const JsonValue& root);

// Trim an over-long sequence from the front, keeping the tail: it holds the most
// recent turn, and dropping older context costs less than dropping the reply the
// model is meant to learn. Returns true if anything was trimmed.
bool truncate_example(Example& example, int max_tokens);

struct DatasetStats {
    size_t examples = 0;
    size_t total_tokens = 0;
    size_t trainable_tokens = 0;
    size_t longest = 0;      // Before truncation
    size_t truncated = 0;
    size_t unknown_tokens = 0;
};

// Truncate, drop sequences with nothing to learn from, and gather stats
std::vector<Example> prepare_examples(const BPETokenizer& tokenizer,
                                      std::vector<Example> examples, int max_tokens,
                                      DatasetStats& stats);

// One-line summary of what came out of prepare_examples
void print_dataset_stats(const DatasetStats& stats, int max_tokens);

#endif // DATASET_H
