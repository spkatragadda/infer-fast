#include "dataset.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <memory>


void append_tokens(Example& example, const std::vector<int>& ids, bool trainable) {
    for (int id : ids) {
        example.tokens.push_back(id);
        example.train_on.push_back(trainable ? 1 : 0);
    }
    if (trainable) example.trainable += static_cast<int>(ids.size());
}

std::vector<std::string> read_text_lines(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open dataset '" + path + "'");
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // Tolerate CRLF files
        if (line.find_first_not_of(" \t") != std::string::npos) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<Example> build_text_examples(const BPETokenizer& tokenizer,
                                         const std::vector<std::string>& lines) {
    std::vector<Example> examples;
    examples.reserve(lines.size());
    for (const auto& line : lines) {
        Example example;
        append_tokens(example, tokenizer.encode(line), true);
        examples.push_back(std::move(example));
    }
    return examples;
}

// The tokenizer splits on whitespace before it merges, so encoding the role
// marker and the content separately and concatenating gives exactly what
// encoding the whole string would - the pieces break at a space either way.
Example build_chat_example(const BPETokenizer& tokenizer, const JsonValue& conversation) {
    Example example;

    for (const JsonValue& message : conversation.array) {
        if (!message.is_object()) {
            throw std::runtime_error("each message must be a JSON object");
        }

        const JsonValue* role = message.find("role");
        const JsonValue* content = message.find("content");
        if (!role || !role->is_string() || !content || !content->is_string()) {
            throw std::runtime_error("each message needs string \"role\" and \"content\" fields");
        }

        if (role->string == "assistant") {
            // The marker is context; what follows it is what the model is being
            // taught to say, so the grading starts right after it
            append_tokens(example, tokenizer.encode("Assistant:"), false);
            append_tokens(example, tokenizer.encode(content->string), true);
        } else if (role->string == "user") {
            append_tokens(example, tokenizer.encode("User: " + content->string), false);
        } else if (role->string == "system") {
            append_tokens(example, tokenizer.encode("System: " + content->string), false);
        } else {
            throw std::runtime_error("unknown role \"" + role->string +
                                     "\"; expected user, assistant or system");
        }
    }
    return example;
}

std::vector<Example> build_chat_examples(const BPETokenizer& tokenizer, const JsonValue& root) {
    if (!root.is_array()) {
        throw std::runtime_error("the top level of the dataset must be a JSON array");
    }

    std::vector<Example> examples;

    // A top level whose elements are objects is one conversation, not many
    bool flat = !root.array.empty() && root.array.front().is_object();
    if (flat) {
        examples.push_back(build_chat_example(tokenizer, root));
        return examples;
    }

    examples.reserve(root.array.size());
    for (const JsonValue& conversation : root.array) {
        if (!conversation.is_array()) {
            throw std::runtime_error("expected an array of conversations, "
                                     "each an array of message objects");
        }
        examples.push_back(build_chat_example(tokenizer, conversation));
    }
    return examples;
}

bool truncate_example(Example& example, int max_tokens) {
    if (max_tokens <= 0 || example.tokens.size() <= static_cast<size_t>(max_tokens)) {
        return false;
    }

    size_t drop = example.tokens.size() - static_cast<size_t>(max_tokens);
    example.tokens.erase(example.tokens.begin(), example.tokens.begin() + drop);
    example.train_on.erase(example.train_on.begin(), example.train_on.begin() + drop);

    example.trainable = 0;
    for (char flag : example.train_on) {
        if (flag) ++example.trainable;
    }
    return true;
}

std::vector<Example> prepare_examples(const BPETokenizer& tokenizer,
                                      std::vector<Example> examples, int max_tokens,
                                      DatasetStats& stats) {
    std::vector<Example> usable;
    usable.reserve(examples.size());

    for (Example& example : examples) {
        stats.longest = std::max(stats.longest, example.tokens.size());
        if (truncate_example(example, max_tokens)) ++stats.truncated;

        // Needs a next token to predict, and something worth grading
        if (example.tokens.size() < 2 || example.trainable == 0) continue;

        stats.total_tokens += example.tokens.size();
        stats.trainable_tokens += static_cast<size_t>(example.trainable);
        for (int id : example.tokens) {
            if (tokenizer.token(id) == "<unk>") ++stats.unknown_tokens;
        }
        usable.push_back(std::move(example));
    }

    stats.examples = usable.size();
    return usable;
}

void print_dataset_stats(const DatasetStats& stats, int max_tokens) {
    std::cout << "Dataset: " << stats.examples << " sequences, " << stats.total_tokens
              << " tokens, " << stats.trainable_tokens << " of them graded" << std::endl;

    std::cout << "Longest sequence: " << stats.longest << " tokens";
    if (stats.truncated > 0) {
        std::cout << " (" << stats.truncated << " truncated to " << max_tokens << ")";
    }
    std::cout << std::endl;

    // A frozen tokenizer can only spell what pretraining taught it
    if (stats.unknown_tokens > 0 && stats.total_tokens > 0) {
        std::cout << "Warning: " << stats.unknown_tokens << " tokens ("
                  << (100.0 * static_cast<double>(stats.unknown_tokens) /
                      static_cast<double>(stats.total_tokens))
                  << "%) fell back to <unk>" << std::endl;
    }
}
