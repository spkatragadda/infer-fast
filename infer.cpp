// Load weights saved by train.cpp and run the model in model.cpp on a string.
//
//   ./infer <model.bin> "some string" [options]

#include "model.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_path;
    std::string prompt;
    int generate = 5;
    bool show_predictions = true;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <model.bin> \"some string\" [options]\n"
              << "\n"
              << "Options:\n"
              << "  --generate N   tokens to continue the prompt with (default 5)\n"
              << "  --quiet        skip the per-position prediction table\n";
}

bool parse_options(int argc, char** argv, Options& opts) {
    if (argc < 3) return false;
    opts.model_path = argv[1];
    opts.prompt = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--quiet") {
            opts.show_predictions = false;
        } else if (flag == "--generate") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --generate\n";
                return false;
            }
            opts.generate = std::stoi(argv[++i]);
        } else {
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
        // 1. Rebuild the architecture from the checkpoint and load its weights
        Model model = Model::load(opts.model_path);
        std::cout << "Loaded " << opts.model_path << ": vocab " << model.config.vocab_size
                  << ", dim " << model.config.embed_dim << ", " << model.config.num_loops
                  << " attention loops" << std::endl;

        // 2. Tokenize the prompt with the vocabulary the model was trained on
        std::vector<int> token_ids = model.tokenizer.encode(opts.prompt);
        if (token_ids.empty()) {
            std::cerr << "Prompt encoded to no tokens\n";
            return 1;
        }

        std::cout << "\nPrompt: \"" << opts.prompt << "\"" << std::endl;
        std::cout << "Tokens: ";
        for (int id : token_ids) {
            std::cout << model.tokenizer.token(id) << " ";
        }
        std::cout << std::endl;

        // 3. One forward pass gives every position's next-token distribution
        if (opts.show_predictions) {
            Matrix probs = model.forward(token_ids);
            std::cout << "\nNext-token prediction at each position:" << std::endl;
            for (int i = 0; i < probs.rows; ++i) {
                const float* row = probs.row(i);
                int best = static_cast<int>(std::max_element(row, row + probs.cols) - row);
                std::cout << "  " << model.tokenizer.token(token_ids[i]) << " -> "
                          << model.tokenizer.token(best) << "\t(p " << row[best] << ")"
                          << std::endl;
            }
        }

        // 4. Feed each prediction back in to continue the string
        if (opts.generate > 0) {
            std::vector<int> ids = model.generate(token_ids, opts.generate);
            std::cout << "\nContinuation: " << model.tokenizer.decode(ids) << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
