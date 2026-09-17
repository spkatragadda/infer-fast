// Train the model in model.cpp on a dataset of strings and save the weights.
//
//   ./train <dataset.txt> <model.bin> [options]
//
// The dataset is plain text: one training string per line, blank lines ignored.

#include "dataset.h"
#include "model.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string dataset_path;
    std::string output_path;
    int epochs = 300;
    int merges = 200;
    int embed_dim = 32;
    int head_dim = 32;
    int num_loops = 4;
    float learning_rate = 0.01f;
    float max_grad_norm = 1.0f;
    unsigned seed = 1234;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <dataset.txt> <model.bin> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --epochs N     passes over the dataset (default 300)\n"
              << "  --merges N     BPE merges to learn (default 200)\n"
              << "  --dim N        embedding and attention width (default 32)\n"
              << "  --loops N      times the shared attention block runs (default 4)\n"
              << "  --lr F         SGD learning rate (default 0.01)\n"
              << "  --clip F       max global gradient norm, 0 disables (default 1.0)\n"
              << "  --seed N       shuffle seed (default 1234)\n";
}

// Parse "--flag value" pairs after the two positional arguments
bool parse_options(int argc, char** argv, Options& opts) {
    if (argc < 3) return false;
    opts.dataset_path = argv[1];
    opts.output_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string flag = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << flag << "\n";
            return false;
        }
        std::string value = argv[++i];

        if (flag == "--epochs") opts.epochs = std::stoi(value);
        else if (flag == "--merges") opts.merges = std::stoi(value);
        else if (flag == "--dim") { opts.embed_dim = std::stoi(value); opts.head_dim = opts.embed_dim; }
        else if (flag == "--loops") opts.num_loops = std::stoi(value);
        else if (flag == "--lr") opts.learning_rate = std::stof(value);
        else if (flag == "--clip") opts.max_grad_norm = std::stof(value);
        else if (flag == "--seed") opts.seed = static_cast<unsigned>(std::stoul(value));
        else {
            std::cerr << "Unknown option: " << flag << "\n";
            return false;
        }
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_options(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        // 1. Load the dataset
        std::vector<std::string> corpus = read_text_lines(opts.dataset_path);
        if (corpus.empty()) {
            std::cerr << "Dataset '" << opts.dataset_path << "' has no usable lines\n";
            return 1;
        }
        std::cout << "Dataset: " << corpus.size() << " lines from " << opts.dataset_path
                  << std::endl;

        // 2. Learn a sub-word vocabulary with Byte Pair Encoding
        BPETokenizer tokenizer;
        tokenizer.train(corpus, opts.merges);
        std::cout << "Vocabulary: " << tokenizer.vocab_size() << " tokens from "
                  << tokenizer.num_merges() << " merges" << std::endl;

        // 3. Encode every line; a sequence needs two tokens to have something to predict
        DatasetStats stats;
        std::vector<Example> sequences =
            prepare_examples(tokenizer, build_text_examples(tokenizer, corpus), 0, stats);
        if (sequences.empty()) {
            std::cerr << "No line encoded to two or more tokens; nothing to train on\n";
            return 1;
        }
        print_dataset_stats(stats, 0);

        // 4. Build the model
        ModelConfig config;
        config.embed_dim = opts.embed_dim;
        config.head_dim = opts.head_dim;
        config.vocab_size = tokenizer.vocab_size();
        config.num_loops = opts.num_loops;
        Model model(config, std::move(tokenizer));

        // 5. Train, visiting the sequences in a fresh order every epoch
        std::cout << "\nTraining " << opts.epochs << " epochs at lr " << opts.learning_rate
                  << ", " << opts.num_loops << " attention loops, clip "
                  << opts.max_grad_norm << ":" << std::endl;

        std::vector<size_t> order(sequences.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 gen(opts.seed);
        int report_every = std::max(1, opts.epochs / 20);

        for (int epoch = 0; epoch < opts.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), gen);

            float epoch_loss = 0.0f;
            for (size_t index : order) {
                epoch_loss += model.train_step(sequences[index].tokens, opts.learning_rate,
                                               opts.max_grad_norm);
            }

            if (epoch % report_every == 0 || epoch == opts.epochs - 1) {
                std::cout << "  epoch " << epoch << "\tloss "
                          << epoch_loss / static_cast<float>(sequences.size()) << std::endl;
            }
        }

        // 6. Save the tokenizer and every weight matrix into one checkpoint
        model.save(opts.output_path);
        std::cout << "\nSaved checkpoint to " << opts.output_path << std::endl;
        std::cout << "Run inference with: ./infer " << opts.output_path << " \"your prompt\""
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
